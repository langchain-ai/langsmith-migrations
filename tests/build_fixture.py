"""Generate an offline test bundle. This does not create any MLflow resources."""
import argparse
from helpers import bundle_fixture
from stack_common import save_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    save_json(args.output, bundle_fixture())
    print('Wrote offline test fixture (not a live MLflow export):', args.output)
