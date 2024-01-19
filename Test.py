from MLMStego import encode,decode,torchInit
import argparse
import torch
import platform
import os

def process_args():
    parser = argparse.ArgumentParser(description='MLMStego for hiding ')

    parser.add_argument('--pathForCoverText', type=str, help='Path to the cover text file.')
    parser.add_argument('--secret', type=str, help='Secret text.')
    parser.add_argument('--language', type=str, default="tr", help='Language parameter (default: "tr").')
    parser.add_argument('--halfWindowSize', type=int, default=5, help='Half window size parameter (default: 10).')
    parser.add_argument('--loopChange', type=int, default=2, help='Loop change parameter (default: 2).')
    parser.add_argument('--loopMod', type=int, default=3, help='Loop mod parameter (default: 3).')
    parser.add_argument('--randomSeed', type=int, default=110001, help='Random seed parameter (default: 110001).')
    parser.add_argument('--saveStegoText', type=bool, default=False, help='Whether to save stego text to a file (default: False).')
    parser.add_argument('--printObtainedSecret', type=bool, default=False, help='Whether to print the obtained secret (default: False).')
    parser.add_argument('--model', type=str, default="dbmdz/bert-base-turkish-cased", help='MLM model to use (default: "dbmdz/bert-base-turkish-cased").')

    device_default = "mps" if platform.system() == "Darwin" else "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=device_default, help='Device to use ("cuda" for GPU, "cpu" for CPU, "mps" for MacOS).')

    args = parser.parse_args()
    return args

def main():
    args = process_args()

    print("Path for cover media:", args.pathForCoverText)
    print("Secret text:", args.secret)
    print("Language:", args.language)
    print("Half window size:", args.halfWindowSize)
    print("Loop change:", args.loopChange)
    print("Loop mod:", args.loopMod)
    print("Random seed:", args.randomSeed)
    print("Model:", args.model)
    print("Device:", args.device)
    print("Save stego text:",args.saveStegoText)
    print("Print obtained secret:",args.printObtainedSecret)

    bertTokenizer, bertModel = torchInit(args.model,args.device)
    coverText = open(args.pathForCoverText,encoding="utf8").read()
    stegoText = encode(args.secret,coverText,args.randomSeed,args.halfWindowSize,args.loopChange,args.loopMod,bertTokenizer,bertModel,args.device)
    obtainedSecret = decode(stegoText,args.halfWindowSize,args.loopMod,bertTokenizer,bertModel,args.device)
    if args.saveStegoText:
        _, extension = os.path.splitext(args.pathForCoverText)
        pathWithoutExtension = os.path.splitext(args.pathForCoverText)[0]
        newPath = pathWithoutExtension + ".hidden" + extension
        open(newPath,"w").write(stegoText)

    if args.printObtainedSecret:
        print("Secret:         ",args.secret)
        print("Obtained Secret:",obtainedSecret)

    if args.secret == obtainedSecret:
        print("Success")

if __name__ == "__main__":
    main()
