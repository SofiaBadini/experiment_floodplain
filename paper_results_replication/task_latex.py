"""Tasks for compiling the paper and presentation(s)."""
import shutil

import pytask
from importlib.metadata import version
from experiment_floodplain.config import BLD, PAPER_DIR

documents = ["paper_results_replication"]

for document in documents:

    if version('pytask-r') == '0.0.7':


        @pytask.mark.latex(
            ["--lualatex", "--interaction=nonstopmode", "--synctex=1", "--cd", "--quiet"]
        )
        @pytask.mark.parametrize(
            "depends_on, produces",
            [
                (
                    PAPER_DIR / f"{document}" / f"{document}.tex",
                    BLD / "latex" / f"{document}.pdf",
                )
            ],
        )
            
        def task_compile_documents():
            pass

        @pytask.mark.depends_on(BLD / "latex" / f"{document}.pdf")
        @pytask.mark.produces(BLD.parent.resolve() / f"{document}.pdf")
        def task_copy_to_root(depends_on, produces):
            """Copy a document to the root directory for easier retrieval."""
            shutil.copy(depends_on, produces)


    else:
        
        from pytask_latex import compilation_steps as cs

        @pytask.mark.latex(
            script=PAPER_DIR / f"{document}" / f"{document}.tex",
            document=BLD / "latex" / f"{document}.pdf",
            compilation_steps=cs.latexmk(
                options=("--lualatex", "--interaction=nonstopmode", "--synctex=1", "--cd"),
            ),
        )
        @pytask.mark.task(id=document)
        def task_compile_document():
            """Compile the document specified in the latex decorator."""

        kwargs = {
            "depends_on": BLD / "latex" / f"{document}.pdf",
            "produces": BLD.parent.resolve() / f"{document}.pdf",
        }

        @pytask.mark.task(id=document, kwargs=kwargs)
        def task_copy_to_root(depends_on, produces):
            """Copy a document to the root directory for easier retrieval."""
            shutil.copy(depends_on, produces)