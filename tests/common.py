# ----------------------------------------------------------------------------
# Copyright (c) 2021-2025 DexForce Technology Co., Ltd.
#
# All rights reserved.
# ----------------------------------------------------------------------------
from unittest import TestLoader
from fnmatch import fnmatchcase


__all__ = ["UnittestMetaclass", "OrderedTestLoader"]


# to learn about the usage of metaclass here: https://www.liaoxuefeng.com/wiki/1016959663602400/1017592449371072
class UnittestMetaclass(type):
    def __new__(cls, name, bases, attrs):
        # add 'attrs_by_writing_order' attribute containing writing order of all attributes and functions
        attrs["attrs_by_writing_order"] = list(attrs.keys())
        return super().__new__(cls, name, bases, attrs)


# By default, TestLoader runs tests in alphabetical order. However, some tests
# need to be executed in the order they are written. This custom loader overrides
# the default sorting behavior to run tests sequentially based on the writing order.
# Note that when both errors and failures occur, errors will be logged first,
# which may differ from the execution order. This is acceptable as it prioritizes
# highlighting errors.
class OrderedTestLoader(TestLoader):
    """This TestLoader will load testFnNames in the code writing order"""

    # copied from getTestCaseNames() of TestLoader and make some modification
    def getTestCaseNames(self, testCaseClass):
        """Return a sorted sequence of method names found within testCaseClass"""

        def shouldIncludeMethod(attrname):
            if not attrname.startswith(self.testMethodPrefix):
                return False
            testFunc = getattr(testCaseClass, attrname)
            if not callable(testFunc):
                return False
            fullName = f"%s.%s.%s" % (
                testCaseClass.__module__,
                testCaseClass.__qualname__,
                attrname,
            )
            return self.testNamePatterns is None or any(
                fnmatchcase(fullName, pattern) for pattern in self.testNamePatterns
            )

        testFnNames = list(
            filter(shouldIncludeMethod, testCaseClass.attrs_by_writing_order)
        )

        return testFnNames
