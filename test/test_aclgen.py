import unittest
import sys
import os
import inspect
import shutil
from cStringIO import StringIO
import filecmp

import aclgen

class Test_AclGen(unittest.TestCase):

  def setUp(self):
    # Capture output during tests.
    self.iobuff = StringIO()
    sys.stdout = self.iobuff
    nullstream = open(os.devnull,'wb')
    sys.stderr = nullstream

  def tearDown(self):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

  def test_smoke_test_generates_successfully_with_no_args(self):
    aclgen.main([])

    expected_output = """writing ./filters/sample_cisco_lab.acl
writing ./filters/sample_gce.gce
writing ./filters/sample_ipset
writing ./filters/sample_juniper_loopback.jcl
writing ./filters/sample_multitarget.jcl
writing ./filters/sample_multitarget.acl
writing ./filters/sample_multitarget.ipt
writing ./filters/sample_multitarget.asa
writing ./filters/sample_multitarget.demo
writing ./filters/sample_multitarget.eacl
writing ./filters/sample_multitarget.bacl
writing ./filters/sample_multitarget.xacl
writing ./filters/sample_multitarget.jcl
writing ./filters/sample_multitarget.acl
writing ./filters/sample_multitarget.ipt
writing ./filters/sample_multitarget.asa
writing ./filters/sample_nsxv.nsx
writing ./filters/sample_packetfilter.pf
writing ./filters/sample_speedway.ipt
writing ./filters/sample_speedway.ipt
writing ./filters/sample_speedway.ipt
writing ./filters/sample_srx.srx
22 filters rendered
"""

    def subtract_list(lhs, rhs):
      return '; '.join([el for el in lhs if el not in rhs])

    expected = expected_output.split("\n")
    actual = self.iobuff.getvalue().split("\n")
    not_in_actual = subtract_list(expected, actual)
    not_in_expected = subtract_list(actual, expected)
    self.assertEqual('', not_in_actual, 'Not in actual: ' + not_in_actual)
    self.assertEqual('', not_in_expected, 'Not in expected: ' + not_in_expected)

  def test_generate_single_policy(self):
    aclgen.main(['-p', 'policies/sample_cisco_lab.pol'])

    expected_output = """writing ./filters/sample_cisco_lab.acl
1 filters rendered
"""
    self.assertEquals(expected_output, self.iobuff.getvalue())

  def test_can_suppress_adding_revision_tags(self):
    curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    fpath = os.path.join(curr_dir, '..', 'filters', 'sample_cisco_lab.acl')

    aclgen.main(['-p', 'policies/sample_cisco_lab.pol'])
    with open(fpath, 'r') as f:
      acl = f.read()
    self.assertIn('$Id: ./filters/sample_cisco_lab.acl $', acl)

    aclgen.main(['-p', 'policies/sample_cisco_lab.pol', '--no-rev-info'])
    with open(fpath, 'r') as f:
      acl = f.read()
    self.assertIn('$Id:$', acl)


class AclGen_filter_name_scenarios(unittest.TestCase):
  """Ensure the output directory structure mirrors the input correctly.

  See aclgen.filter_name for notes."""

  def test_file_in_base_directory_is_output_to_output_dir(self):
    s = aclgen.filter_name('/policy', '/policy/x.txt', '.out', '/output')
    self.assertEqual(s, '/output/x.out')

  def test_file_in_subdirectory_is_output_to_subdirectory(self):
    s = aclgen.filter_name('/policy', '/policy/subdir/x.txt', '.out', '/output')
    self.assertEqual(s, '/output/subdir/x.out')

  def test_file_in_nested_subdir(self):
    s = aclgen.filter_name('/policy', '/policy/sub/dir/x.txt', '.out', '/output/here')
    self.assertEqual(s, '/output/here/sub/dir/x.out')

  def test_relative_directory_structure_mirrored(self):
    s = aclgen.filter_name('../policy', '../policy/subdir/x.txt', '.out', '/output')
    self.assertEqual(s, '/output/subdir/x.out')

  def test_empty_source_dir_ok(self):
    s = aclgen.filter_name('', 'subdir/x.txt', '.out', '/output')
    self.assertEqual(s, '/output/subdir/x.out')

  def test_base_dir_must_match_start_of_source_file(self):
    self.assertRaises(ValueError, aclgen.filter_name, 'A', 'B/C', 'suff', 'O')


class AclGen_Characterization_Tests(unittest.TestCase):
  """Ensures base functionality works.  Uses data in
  characterization_data subfolder."""

  def setUp(self):
    # Capture output during tests.
    self.iobuff = StringIO()
    sys.stderr = sys.stdout = self.iobuff

    curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    self.test_dir = os.path.join(curr_dir, 'characterization_data')
    self.output_dir = os.path.join(self.test_dir, 'filters_actual')
    self.empty_output_dir(self.output_dir)

  def empty_output_dir(self, d):
    entries = [ os.path.join(d, f) for f in os.listdir(d)]
    for f in [e for e in entries if os.path.isfile(e)]:
      os.remove(f)
    for d in [e for e in entries if os.path.isdir(e)]:
      shutil.rmtree(d)

  def tearDown(self):
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

  def test_characterization(self):
    def make_path(x): return os.path.join(self.test_dir, x)
    def_dir, pol_dir, expected_dir = map(make_path, ('def', 'policies', 'filters_expected'))
    aclgen.main(['-d', def_dir, '--poldir', pol_dir, '-o', self.output_dir, '--no-rev-info'])

    dircmp = filecmp.dircmp(self.output_dir, expected_dir)
    self.assertEquals([], dircmp.left_only, 'missing {0} in filters_expected'.format(dircmp.left_only))
    self.assertEquals([], dircmp.right_only, 'missing {0} in filters_actual'.format(dircmp.right_only))
    self.assertEquals([], dircmp.diff_files)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
