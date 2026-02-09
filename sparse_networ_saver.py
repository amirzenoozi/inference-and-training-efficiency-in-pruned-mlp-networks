import gzip


def pth_to_gz(input_file):
    with open(input_file, 'rb') as f_in:
        with gzip.open(f'{input_file}.gz', 'wb') as f_out:
            f_out.writelines(f_in)


def gz_to_pth(input_file):
    with gzip.open(input_file, 'rb') as f_in:
        with open('output_file.pth', 'wb') as f_out:
            f_out.writelines(f_in)


pth_to_gz('best_model.pth')
gz_to_pth('best_model.pth.gz')
# Example usage
