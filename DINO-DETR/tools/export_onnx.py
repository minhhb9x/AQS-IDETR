import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch
import torch.nn as nn
import numpy as np

from util.slconfig import DictAction, SLConfig


def get_args_parser():
    parser = argparse.ArgumentParser('Export DINO model to ONNX', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True,
                        help='Path to the config file')
    parser.add_argument('--options', '-o', nargs='+', action=DictAction)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='model.onnx')
    parser.add_argument('--input_size', nargs=2, type=int, default=[640, 640])
    parser.add_argument('--opset_version', type=int, default=11)
    parser.add_argument('--dynamic_axes', action='store_true')
    parser.add_argument('--verify_model', action='store_true')
    return parser


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def load_model(args, checkpoint_path):
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
    model, criterion, postprocessors = build_model_main(args)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    new_state_dict = {k[7:] if k.startswith('module.') else k: v
                      for k, v in model_state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model, postprocessors, args


def export_to_onnx(model, inputs, output_path, opset_version=11, dynamic_axes=False):
    images, orig_target_sizes = inputs
    input_names = ['images', 'orig_target_sizes']
    output_names = ['boxes', 'scores', 'labels']

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'orig_target_sizes': {0: 'batch'},
            'boxes': {0: 'batch'},
            'scores': {0: 'batch'},
            'labels': {0: 'batch'}
        }

    print('Dynamic axes:', dynamic_axes_dict)
    torch.onnx.export(
        model, (images, orig_target_sizes), output_path,
        input_names=input_names, output_names=output_names,
        export_params=True, opset_version=opset_version,
        do_constant_folding=True, dynamic_axes=dynamic_axes_dict,
        verbose=False
    )
    print(f"✓ Model exported to {output_path}")


def verify_onnx_model(onnx_path, images, orig_target_sizes, pytorch_model):
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(onnx_path))
    print("ONNX model is valid")

    ort_session = ort.InferenceSession(onnx_path)
    with torch.no_grad():
        pt_boxes, pt_scores, pt_labels = pytorch_model(images, orig_target_sizes)

    ort_inputs = {
        ort_session.get_inputs()[0].name: images.cpu().numpy(),
        ort_session.get_inputs()[1].name: orig_target_sizes.cpu().numpy()
    }
    ort_boxes, ort_scores, ort_labels = ort_session.run(None, ort_inputs)

    print("Boxes match:", np.allclose(pt_boxes.cpu().numpy(), ort_boxes, rtol=1e-3, atol=1e-5))
    print("Scores match:", np.allclose(pt_scores.cpu().numpy(), ort_scores, rtol=1e-3, atol=1e-5))
    print("Labels match:", np.allclose(pt_labels.cpu().numpy().astype(np.float32),
                                       ort_labels.astype(np.float32), rtol=1e-3, atol=1e-5))


def main():
    parser = argparse.ArgumentParser('Export DINO model to ONNX', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device('cuda')
    args.device = device
    model, postprocessors, model_args = load_model(args, args.checkpoint_path)
    postprocessors = postprocessors['bbox'] # Use only bbox postprocessor
    class ExportWrapper(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model
            self.post = postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.post(outputs, orig_target_sizes)

    export_model = ExportWrapper(model, postprocessors).eval().to(device)
    
    height, width = args.input_size
    images = torch.randn(1, 3, height, width, device=device)
    orig_target_sizes = torch.tensor([[height, width]] * 1, device=device)

    with torch.no_grad():
        out = export_model(images, orig_target_sizes)   # chạy ở Python, không export
    print("Forward OK", type(out))

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    export_to_onnx(export_model, (images, orig_target_sizes),
                   args.output_path, args.opset_version, args.dynamic_axes)

    if args.verify_model:
        verify_onnx_model(args.output_path, images, orig_target_sizes, export_model)


if __name__ == '__main__':
    main()
