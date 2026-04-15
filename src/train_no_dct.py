"""Entry point untuk training image-only (tanpa DCT)."""

import os
import sys

from train import main


if __name__ == "__main__":
	os.environ["USE_DCT"] = "0"
	sys.exit(main())
