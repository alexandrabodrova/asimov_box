"""
Convert text file with prompts to pickle file.

Split data with '--0000--'

"""
import argparse
import pickle


def main(args):

    # Load text file
    with open(args.data_file, 'r') as f:
        data = f.read()

    # Split data
    data_all = data.split('--0000--')
    data_all = [{'prompt': d} for d in data_all]

    # Save data
    with open(args.save_path, 'wb') as f:
        pickle.dump(data_all, f)
    print('Saved to {}'.format(args.save_path))
    print('Number of data: {}'.format(len(data_all)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", help="data file path", default='', type=str
    )
    parser.add_argument(
        "--save_path", help="data file path", default='', type=str
    )
    args = parser.parse_args()
    main(args)