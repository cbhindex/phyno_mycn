"""Enable ``python -m pheno_mycn`` as an alias for the ``pheno-mycn`` CLI."""

from pheno_mycn.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
