"""Match genes, apply the recorded QC rule and prepare the preserved split."""
import argparse
from _run import run_script

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_script('prepare_whole_blood.py')
