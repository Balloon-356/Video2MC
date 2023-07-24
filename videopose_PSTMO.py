import os
import time

from common.arguments import parse_args
from common.camera import *
from common.generators import *
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path
from common.inference_3d import *

from model.block.refine import refine
from model.stmo import Model

import HPE2keyframes as Hk 

# from joints_detectors.openpose.main import generate_kpts as open_pose


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}

add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()


time0 = ckpt_time()


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        'hr_pose': get_hr_pose,
        # 'open_pose': open_pose
    }

    assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'

    return detector_map[detector_name]()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 14, 15, 16]


def main(args):
    detector_2d = get_detector_2d(args.detector_2d)

    assert detector_2d, 'detector_2d should be in ({alpha, hr, open}_pose)'

    # 2D kpts loads or generate
    #args.input_npz = './outputs/alpha_pose_skiing_cut/skiing_cut.npz'
    if not args.input_npz:
        video_name = args.viz_video
        keypoints = detector_2d(video_name)
    else:
        npz = np.load(args.input_npz)
        keypoints = npz['kpts']  # (N, 17, 2)

    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])

    # normlization keypoints  Suppose using the camera parameter
    keypoints = normalize_screen_coordinates(keypoints[..., :2], w=1000, h=1002)

    # model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
    #                           dense=args.dense)

    model = {}
    model['trans'] = Model(args).cuda()


    # if torch.cuda.is_available():
    #     model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load data spends {:.2f} seconds'.format(ckpt))

    # load trained model
    # chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    # print('Loading checkpoint', chk_filename)
    # checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)  # 把loc映射到storage
    # model_pos.load_state_dict(checkpoint['model_pos'])

    model_dict = model['trans'].state_dict()

    no_refine_path = "checkpoint/PSTMOS_no_refine_48_5137_in_the_wild.pth"
    pre_dict = torch.load(no_refine_path)
    for key, value in pre_dict.items():
        name = key[7:]
        model_dict[name] = pre_dict[key]
    model['trans'].load_state_dict(model_dict)


    ckpt, time2 = ckpt_time(time1)
    print('-------------- load 3D model spends {:.2f} seconds'.format(ckpt))

    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = args.frames
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    print('Rendering...')
    input_keypoints = keypoints.copy()
    print(input_keypoints.shape)
    # gen = UnchunkedGenerator(None, None, [input_keypoints],
    #                          pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
    #                          kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
    # test_data = Fusion(opt=args, train=False, dataset=dataset, root_path=root_path, MAE=opt.MAE)
    # test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1,
    #                                               shuffle=False, num_workers=0, pin_memory=True)
    #prediction = evaluate(gen, model_pos, return_predictions=True)

    gen = Evaluate_Generator(128, None, None, [input_keypoints], args.stride,
                             pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation, shuffle=False,
                             kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    prediction = val(args, gen, model)

    # save 3D joint points
    # np.save(f'outputs/test_3d_{args.video_name}_output.npy', prediction, allow_pickle=True)

    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    prediction = camera_to_world(prediction, R=rot, t=0)

    # We don't have the trajectory, but at least we can rebase the height
    prediction[:, :, 2] -= np.min(prediction[:, :, 2])

    output_dir_dict = {}
    npy_filename = f'output_3Dpose_npy/{args.video_name}.npy'
    output_dir_dict['npy'] = npy_filename
    np.save(npy_filename, prediction, allow_pickle=True)

    anim_output = {'Ours': prediction}
    input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

    ckpt, time3 = ckpt_time(time2)
    print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

    if not args.viz_output:
        args.viz_output = 'outputs/alpha_result.mp4'

    from common.visualization import render_animation
    render_animation(input_keypoints, anim_output,
                     Skeleton(), 25, args.viz_bitrate, np.array(70., dtype=np.float32), args.viz_output,
                     limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                     input_video_path=args.viz_video, viewport=(1000, 1002),
                     input_video_skip=args.viz_skip)

    ckpt, time4 = ckpt_time(time3)
    print('total spend {:2f} second'.format(ckpt))

    return output_dir_dict


def inference_video(video_path, detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()

    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    args.video_name = basename[:basename.rfind('.')]
    args.viz_video = video_path
    args.viz_output = f'output_videos/{args.video_name}.mp4'
    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        output_dir_dict = main(args)

    output_dir_dict["output_videos"] = args.viz_output
    output_dir_dict["video_name"] = args.video_name
    return output_dir_dict


if __name__ == '__main__':

    files = os.listdir('./input_videos')
    FPS_mine_imator = 30
    for file in files:
        output_dir_dict = inference_video(os.path.join('input_videos', file), 'alpha_pose')
        Hk.hpe2keyframes(output_dir_dict['npy'], FPS_mine_imator, f"output_miframes/{output_dir_dict['video_name']}.miframes")
