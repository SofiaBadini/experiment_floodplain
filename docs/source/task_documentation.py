import shutil
import subprocess
from pathlib import Path
from importlib.metadata import version

import pytask

from experiment_floodplain.config import SRC, BLD, DOCS


#@pytask.mark.depends_on(
#    list(Path(__file__).parent.glob("*.rst")) + [DOCS / "source" / "conf.py"]
#)
@pytask.mark.depends_on(
    list((DOCS / "source").glob("*.rst")) + [DOCS / "source" / "conf.py"]
)
@pytask.mark.parametrize(
    "builder, produces",
    [
        ("latexpdf", BLD / "documentation" / "latex" / "project_documentation.pdf"),
        ("html", (BLD / "documentation" / "html").rglob("*.*")),
    ],
)
def task_build_documentation(builder, produces):
    subprocess.run(
        [
            "sphinx-build",
            "-M",
            builder,
            DOCS.joinpath("source").as_posix(),
            BLD.joinpath("documentation").as_posix(),
        ]
    )

if version('pytask-r') == '0.0.7':

    @pytask.mark.depends_on(BLD / "documentation" / "latex" / "project_documentation.pdf")
    @pytask.mark.produces(BLD.parent.resolve() / "project_documentation.pdf")
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)


else:

    kwargs = {
        "depends_on": BLD / "documentation" / "latex" / "project_documentation.pdf",
        "produces": BLD.parent.resolve() / "project_documentation.pdf"
    }

    @pytask.mark.task(kwargs=kwargs)
    def task_copy_to_root(depends_on, produces):
        """Copy a document to the root directory for easier retrieval."""
        shutil.copy(depends_on, produces)
