"""The sandbox upload must include every local runtime import dependency."""
import ast
from pathlib import Path
import unittest


class AdvisorSandboxPackageTest(unittest.TestCase):
    def test_uploaded_modules_include_relative_imports(self):
        package = Path(__file__).resolve().parents[1] / 'src/autoresearch_budget'
        adapter = ast.parse((package/'harbor_mini_agent.py').read_text())
        cls = next(n for n in adapter.body if isinstance(n,ast.ClassDef) and n.name=='AdvisoryMiniSweAgent')
        assignment = next(n for n in cls.body if isinstance(n,ast.Assign)
                          and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in n.targets))
        files = set(ast.literal_eval(assignment.value))
        self.assertIn('advisory_agent.py',files)
        for filename in files:
            tree=ast.parse((package/filename).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                    dependency=node.module.split('.')[0]+'.py'
                    if (package/dependency).is_file():
                        with self.subTest(module=filename,dependency=dependency):
                            self.assertIn(dependency,files)
