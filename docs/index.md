# Py2DTM

Two-dimensional template-matching (2DTM) for *in situ* structural biology implemented in Python.
This package reflects most of the functionality described in Lucas, *et al.* (2021)[^1].

## Installation

Pre-packaged versions of Py2DTM are released on the Python Package Index (PyPI).
We target Linux operating systems on Python 3.9 and above for these releases, and the PyTorch GPU acceleration backend is only tested against NVIDIA GPUs.
With these caveats in mind, the package can be installed using pip:

```bash
pip install py2dtm
```

We also recommend you install the package in a virtual environment (such as [conda](https://docs.conda.io/en/latest/)) to avoid conflicts with other packages.

### Installing from Source

To install the package from source, first clone the repository and install the package using pip:

```bash
git clone https://github.com/jdickerson95/tt2DTM.git
cd tt2DTM
pip install .
```

### For Developers

For developers interested in contributing to the package, we recommend installing the package in an editable configuration with the necessary development dependencies:

```bash
git clone https://github.com/jdickerson95/tt2DTM.git
cd tt2DTM
pip install -e '.[dev,test,docs]'
```

See the [Contributing](#contributing) section for more information on how to contribute to the package.

## Basic Usage

A minimally working example of running the `match_template` program in a python script is shown below.
Please see the examples page for more extensive explanations and demonstrations on how to configure, use, and extend the package.

```python
from tt2dtm.pydantic_models import MatchTemplateManager
from tt2dtm.pydantic_models import MatchTemplateResult
from tt2dtm.pydantic_models import OpticsGroup
from tt2dtm.pydantic_models import DefocusSearchConfig
from tt2dtm.pydantic_models import OrientationSearchConfig

# Microscope imaging parameters
my_optics_group = OpticsGroup(
    label="my_optics_group",
    pixel_size=1.2,    # In Angstroms
    voltage=300,       # In kV
    defocus_u=5100.0,  # In Angstroms
    defocus_v=4900.0,  # In Angstroms
    astigmatism_angle=0.0,  # In degrees
)

# Relative defocus planes to search across
df_search_config = DefocusSearchConfig(
    label="defocus_search",
    min_defocus=1000,   # In Angstroms, relative
    max_defocus=-1000,  # In Angstroms, relative
    step_size=200.0,    # In Angstroms
)

# Orientation sampling of SO(3) space
orientation_search_config = OrientationSearchConfig()

# Where to save the output results
mt_result = MatchTemplateResult(
    allow_file_overwrite=True,
    mip_path="/path/to/output_mip.mrc",
    scaled_mip_path="/path/to/output_scaled_mip.mrc",
    correlation_average_path="/path/to/output_correlation_average.mrc",
    correlation_variance_path="/path/to/output_correlation_variance.mrc",
    orientation_psi_path="/path/to/output_orientation_psi.mrc",
    orientation_theta_path="/path/to/output_orientation_theta.mrc",
    orientation_phi_path="/path/to/output_orientation_phi.mrc",
    relative_defocus_path="/path/to/output_relative_defocus.mrc",
    pixel_size_path="/path/to/output_pixel_size.mrc",
)

mt_manager = MatchTemplateManager(
    micrograph_path="/path/to/2D_image.mrc",
    template_volume_path="/path/to/template_volume.mrc",
    optics_group=my_optics_group,
    defocus_search_config=df_search_config,
    orientation_search_config=orientation_search_config,
    match_template_result=mt_result,
    # pixel_size_search_config
    # preprocessing_filters
    # computational_config
)


def main():
    # Batch size helps control GPU memory usage
    mt_manager.run_match_template(orientation_batch_size=8)

if __name__ == "__main__":
    main()
```

## Documentation and Examples

TODO

## Theory

TODO

## API

TODO: Get some autodocs to parse the docstrings and generate API documentation.

## Contributing
We encourage contributions to this package from the broader cryo-EM/ET and structural biology communities.
Py2DTM is configured with a set of development dependencies to help contributors maintain code quality and consistency.
See the [Installation -- For Developers](#for-developers) section for instructions on how to install these dependencies.

### Using `pre-commit`
The `pre-commit` package is used to run a set of code quality checks and auto-formatters on the codebase.
If this is your first time installing the package, you will need to install the pre-commit hooks:

```bash
pre-commit install--install-hooks
```

After staging changes, but before making a commit, you can run the pre-commit checks with:

```bash
pre-commit run
```

This will go through the staged files, check that all the changed code adheres to the style guidelines, and auto-format the code where necessary.
If all the tests pass, you can commit the changes.

### Running Tests
Py2DTM uses the `pytest` package for running tests.
To run the tests, simply run the following command from the root directory of the repository:

```bash
pytest
```

Note that we are still working on expanding the unit tests to cover more of the package, but we ask that any new code contributions include tests where appropriate.

### Building Documentation
The documentation for Py2DTM is built using [MkDocs](https://www.mkdocs.org) and [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) for generating the documentation site.
If you've installed the package with the optional `docs` dependencies, you can build the documentation site with the following command:

```bash
mkdocs build
mkdocs serve
```

The first command will construct the HTML files for the documentation site, and the second command will start a local server (at `127.0.0.1:8000`) to view the site.

## License

The code in this repository is licensed under the **BSD 3-Clause License**. See the [LICENSE](LICENSE) file for full details.

## References

[^1]: Lucas BA, Himes BA, Xue L, Grant T, Mahamid J, Grigorieff N. Locating macromolecular assemblies in cells by 2D template matching with cisTEM. Elife. 2021 Jun 11;10:e68946. doi: 10.7554/eLife.68946. PMID: 34114559; PMCID: PMC8219381.