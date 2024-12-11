from argparse import ArgumentParser

from common.sb3.util import str2bool

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--w', type=int)
    parser.add_argument('--use_seg_ang', type=str2bool, default=False)
    parser.add_argument('--env_id', type=str)
    args = parser.parse_args()
    env_name_tail = "MinEffort"
    if args.env_id is not None:
        env_name_tail = args.env_id
    if args.use_seg_ang:
        env_name_tail += "_segAng"
    name_tail = f"_{env_name_tail}_ptb1to7/softPD500100_{args.w+1}vs{99-args.w}_hardLim"
    print(name_tail)
