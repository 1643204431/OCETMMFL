import argparse

parser = argparse.ArgumentParser()
# general
parser.add_argument('-go', "--goal", type=str, default="test",
                    help="The goal for this experiment")
parser.add_argument('-dev', "--device", type=str, default="cuda",
                    choices=["cpu", "cuda"])
parser.add_argument('-did', "--device_id", type=str, default="0")
parser.add_argument('-data', "--dataset", type=str, default="har")
parser.add_argument('-nb', "--num_classes", type=int, default=6)
parser.add_argument('-lbs', "--batch_size", type=int, default=16)
parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                    help="Local learning rate")
parser.add_argument('-gr', "--global_rounds", type=int, default=1000)
parser.add_argument('-ls', "--local_steps", type=int, default=5)

parser.add_argument('-nc', "--num_clients", type=int, default=3,
                    help="Total number of clients")

parser.add_argument('-sfn', "--save_folder_name", type=str, default='models')
args = parser.parse_args()