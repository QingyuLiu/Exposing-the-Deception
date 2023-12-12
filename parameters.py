import argparse
def parse_args():

    parser = argparse.ArgumentParser()

    # general arguments
    parser.add_argument("--name", default="train_model", type=str, help="Specify name of the model")
    parser.add_argument("--gpu_num", default="0,1,2,3", type=str,
                        help="gpu number")

    # arguments for train
    parser.add_argument('--model', default="resnet", type=str, help="choose backbone model")
    parser.add_argument("--epoch", default=10, type=int, help="epoch of training")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay of training")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate of training")
    parser.add_argument("--bs", default=256, type=int, help="batch size of training")
    parser.add_argument("--test_bs", default=1000, type=int, help="batch size of training")
    parser.add_argument("--num_workers", default=12, type=int, help="num workers")

    # arguments for loss
    parser.add_argument("--lil_loss", default=True, type=bool, help="if local information loss")
    parser.add_argument("--gil_loss", default=True, type=bool, help="if global information loss")
    parser.add_argument('--temperature', type=float, default=1.5, help="the temperature used in knowledge distillation")
    parser.add_argument('--mi_calculator', default="kl", type=str, help="mutual information calculation method")
    parser.add_argument('--scales', default=[1,2,10], type=list, help="multiple losses weights")
    parser.add_argument('--balance_loss_method', default='auto', type=str, help="balance multiple losses method")

    #model parameters
    parser.add_argument("--num_LIBs", default=4, type=int, help="the number of Local Information Block")
    parser.add_argument("--resume_model",
                        default="",
                        type=str,
                        help="Path of resume model")

    # arguments for test
    parser.add_argument("--test", default=False, type=bool,help="Test or not")

    # save models
    parser.add_argument("--save_model", default=True, type=bool,help="whether save models or not")
    parser.add_argument("--save_path", default="output", type=str,
                        help="Path of test file, work when test is true")

    # dataset
    parser.add_argument("--size", default=224, type=int,
                        help="Specify the size of the input image, applied to width and height")
    parser.add_argument('--dataset', default="FF++", type=str, help="dataset txt path")#
    # 'Face2Face','Deepfakes','FaceSwap','NeuralTextures', Celeb-DF-v2, DFDC-Preview,DFDC,FF++_c23,DeeperForensics-1.0
    parser.add_argument("--mixup", default=True, type=bool,help="mix up or not")
    parser.add_argument("--alpha", default=0.5, type=float, help="mix up alpha")

    return parser.parse_args()

