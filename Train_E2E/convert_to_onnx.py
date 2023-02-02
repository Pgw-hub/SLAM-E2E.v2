import os
import sys
from tkinter import NE; sys.path.append(os.path.dirname(__file__))
import onnx
import torch
from onnxsim.onnx_simplifier import simplify
from model import NetworkNvidia
import argparse

def convert_to_onnx(args):
    ckpt_path = args.ckpt

    state_dict = torch.load(ckpt_path)

    model = NetworkNvidia().cuda(0)
    model.load_state_dict(state_dict["state_dict"])
    model.eval()

    input_tensor_dummy = torch.randn(2,3,70,320,requires_grad = False).cuda(0)
    
    onnx_filename = os.path.splitext(os.path.basename(ckpt_path))[0] + ".onnx"
    # out_onnx_path = os.path.join(os.path.dirname(ckpt_path), onnx_filename)
    out_onnx_path = "/home/ORB_SLAM3_juno/Trained_Model/" + onnx_filename

    torch.onnx.export(model,
                    args = (input_tensor_dummy),
                    f = out_onnx_path,
                    export_params = True,
                    opset_version=8,
                    do_constant_folding = True,
                    input_names = ["input"],
                    output_names = ["output"])


    model_onnx_simplified, check_ok = simplify(out_onnx_path,
                                            check_n = 3,
                                            perform_optimization = True,
                                            skip_fuse_bn = False,
                                            skip_shape_inference=False,
                                            dynamic_input_shape=False)

    onnx.save(model_onnx_simplified, out_onnx_path.replace(".onnx", ".simplified.onnx"))
    
    test = 1

def new_func():
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Convert to onnx")
    parser.add_argument("--ckpt",type = str)

    args = parser.parse_args()
    convert_to_onnx(args)


