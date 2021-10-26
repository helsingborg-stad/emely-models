from parlai.core.opt import Opt
from parlai.scripts.torchscript import export_emely
from pathlib import Path

"""
This is a script that scripts the emely-model with custom settings for
    - Inference type
    - Beamsize
    - Quantization
"""
def main(loadpath, savepath, inference, beamsize, quantize):
    
    opt_path = Path(loadpath) / 'model.opt'
    opt = Opt.load(opt_path)

    # Change opts
    opt['skip_generation'] = False
    opt['init_model'] = (Path(loadpath) / 'model').as_posix()
    opt['no_cuda'] = True  # Cloud run doesn't offer gpu support

    # Inference options
    opt['inference'] = inference
    opt['beam_size'] = beamsize

    opt["scripted_model_file"] = savepath
    opt["model_file"] = opt["init_model"]
    opt["temp_separator"] = "__space__"
    opt["bpe_add_prefix_space"] = False
    
    _,_ = export_emely(opt,quantize)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Script Emely using torchscript')
    parser.add_argument('--loadpath', metavar='path', required=True,
                        help='The path to the trained parlai model directory')
    parser.add_argument('--savepath', metavar='savepath', required=True,
                        help='The path where to save the scripted module')
    parser.add_argument('--inference', metavar='inference', required=False, default="beam",
                        help='Type of inference algorithm ("beam" or "greedy"')
    parser.add_argument('--beamsize', metavar='beamsize', required=False, default=10, type=int,
                        help='The beam size to use in beam-search')
    parser.add_argument('--quantize', metavar='quantize', required=False, default=False, type=bool,
                        help='To quantize or not (bool)')
    args = parser.parse_args()
    main(args.loadpath, args.savepath, args.inference, args.beamsize, args.quantize)