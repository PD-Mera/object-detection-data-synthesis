import argparse

from mdetsyn import run_synthesis, create_args

if __name__ == "__main__":
    args = create_args()
    run_synthesis(args)
