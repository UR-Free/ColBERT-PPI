"""Train on the supplied contact-labelled examples."""
from colbert_ppi.options import run_script


def run(args):
    from colbert_ppi.training import train
    train(args)


if __name__ == "__main__":
    run_script("train", run)
