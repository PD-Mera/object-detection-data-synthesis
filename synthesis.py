import argparse

from mdetsyn import run_synthesis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object Detection Data Synthesis')
    parser.add_argument('--backgrounds', default='./backgrounds', type=str, help='Path to background images directory')
    parser.add_argument('--objects', default='./objects', type=str, help='Path to objects images directory')
    parser.add_argument('--savename', default='./synthesis', type=str, help='Path to save synthesis images directory')
    parser.add_argument('--number', default=1, type=int, help='Number of generate labels for each class')
    parser.add_argument('--class_mapping', default='./class_mapping.json', type=str, help='Path to class mapping file')
    parser.add_argument('--class_txt', default=None, type=str, help='Path to classes.txt file')

    args = parser.parse_args()

    run_synthesis(args.backgrounds,
                  args.objects,
                  args.savename,
                  args.number,
                  args.class_mapping,
                  args.class_txt)
