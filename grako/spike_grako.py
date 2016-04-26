"""Parses the generic policy files and return a policy object for acl rendering.
"""

import grako
from parse_acl_base import ACLParser
from parse_acl_base import ACLSemantics

from policy import Header, Term
import sys


def load_policy(data):
    """Loads a policy object from Grako data"""
    print 'to do, load_policy'
    # h = Header()
    # d = data['header']

    # refs = {
    #     'target': h.target,
    #     'comment': h.comment
    # }
    
    # # print d[0].dump()
    # attrs = d['attributes']
    # # print attrs.dump()
    # for i in range(0, len(attrs)):
    #     k = attrs[i]['key']
    #     v = attrs[i]['value']
    #     refs[k].append(v)

    # # print h
    
    # # for t in data['header']['attributes']:
    # #     print t
    # #     print t['name']
    # #     print t['value']

    # terms = []
    # tdata = data['terms']
    # for i in range(0, len(tdata)):
    #     # print 'loading term {0}...'.format(i)
    #     td = tdata[i]
    #     # print td['name']
    #     term = Term()
    #     term.name = td['name']
    #     attrs = td['attributes']
    #     # print attrs.dump()
    #     for i in range(0, len(attrs)):
    #         k = attrs[i]['key']
    #         v = attrs[i]['value']
    #         print '{0} => {1}'.format(k, v)
    #         # print k
        

def try_parse(parser, s):
    print '--------------------'
    print 'FILE: ' + s
    with open('../test/characterization_data/policies/{0}'.format(s), 'r') as f:
        data = f.read()
    d = parser.parse(data, rule_name='START')
    print d
    p = load_policy(d)
    print 'ok'


class MyACLSemantics(ACLSemantics):
    """Overrides."""

    def eol(self, ast): pass

    def delim(self, ast): pass

    def openbracket(self, ast): pass

    def closebracket(self, ast): pass

    def target_spec(self, ast):
        return [ast[0], ' '.join(ast[1])]

    def header(self, ast):
        print 'entering header with attributes{0}'.format(ast['attributes'])
        return ast

    def term(self, ast):
        print 'entering term {0}'.format(ast['name'])
        print 'with attributes {0}'.format(ast['attributes'])
        return ast

def main():
    parser = ACLParser(semantics = MyACLSemantics())

    files = """sample_cisco_lab.pol
sample_gce.pol
sample_ipset.pol
sample_juniper_loopback.pol
sample_multitarget.pol
sample_nsxv.pol
sample_packetfilter.pol
sample_speedway.pol
sample_srx.pol
test_pyparsing_spike.pol"""
    for f in files.split('\n'):
        try_parse(parser, f)

if __name__ == '__main__':
    main()


sys.exit(0)

# ==================

import datetime


# This could be a VarType object, but I'm keeping it as it's class
# b/c we're almost certainly going to have to do something more exotic with
# it shortly to account for various rendering options like default iptables
# policies or output file names, etc. etc.
class Target(object):
  """The type of acl to be rendered from this policy file."""

  def __init__(self, target):
    self.platform = target[0]
    if len(target) > 1:
      self.options = target[1:]
    else:
      self.options = None

  def __str__(self):
    return self.platform

  def __eq__(self, other):
    return self.platform == other.platform and self.options == other.options

  def __ne__(self, other):
    return not self.__eq__(other)


# Lexing/Parsing starts here
tokens = (
    'ACTION',
    'ADDR',
    'ADDREXCLUDE',
    'COMMENT',
    'COUNTER',
    'DADDR',
    'DADDREXCLUDE',
    'DPFX',
    'DPORT',
    'DINTERFACE',
    'DQUOTEDSTRING',
    'DTAG',
    'ETHER_TYPE',
    'EXPIRATION',
    'FRAGMENT_OFFSET',
    'HEADER',
    'ICMP_TYPE',
    'INTEGER',
    'LOGGING',
    'LOSS_PRIORITY',
    'OPTION',
    'OWNER',
    'PACKET_LEN',
    'PLATFORM',
    'PLATFORMEXCLUDE',
    'POLICER',
    'PORT',
    'PRECEDENCE',
    'PRINCIPALS',
    'PROTOCOL',
    'PROTOCOL_EXCEPT',
    'QOS',
    'ROUTING_INSTANCE',
    'SADDR',
    'SADDREXCLUDE',
    'SINTERFACE',
    'SPFX',
    'SPORT',
    'STAG',
    'STRING',
    'TARGET',
    'TERM',
    'TIMEOUT',
    'TRAFFIC_TYPE',
    'VERBATIM',
)

literals = r':{},-'
t_ignore = ' \t'

reserved = {
    'action': 'ACTION',
    'address': 'ADDR',
    'address-exclude': 'ADDREXCLUDE',
    'comment': 'COMMENT',
    'counter': 'COUNTER',
    'destination-address': 'DADDR',
    'destination-exclude': 'DADDREXCLUDE',
    'destination-interface': 'DINTERFACE',
    'destination-prefix': 'DPFX',
    'destination-port': 'DPORT',
    'destination-tag': 'DTAG',
    'ether-type': 'ETHER_TYPE',
    'expiration': 'EXPIRATION',
    'fragment-offset': 'FRAGMENT_OFFSET',
    'header': 'HEADER',
    'icmp-type': 'ICMP_TYPE',
    'logging': 'LOGGING',
    'loss-priority': 'LOSS_PRIORITY',
    'option': 'OPTION',
    'owner': 'OWNER',
    'packet-length': 'PACKET_LEN',
    'platform': 'PLATFORM',
    'platform-exclude': 'PLATFORMEXCLUDE',
    'policer': 'POLICER',
    'port': 'PORT',
    'precedence': 'PRECEDENCE',
    'principals': 'PRINCIPALS',
    'protocol': 'PROTOCOL',
    'protocol-except': 'PROTOCOL_EXCEPT',
    'qos': 'QOS',
    'routing-instance': 'ROUTING_INSTANCE',
    'source-address': 'SADDR',
    'source-exclude': 'SADDREXCLUDE',
    'source-interface': 'SINTERFACE',
    'source-prefix': 'SPFX',
    'source-port': 'SPORT',
    'source-tag': 'STAG',
    'target': 'TARGET',
    'term': 'TERM',
    'timeout': 'TIMEOUT',
    'traffic-type': 'TRAFFIC_TYPE',
    'verbatim': 'VERBATIM',
}


# disable linting warnings for lexx/yacc code
# pylint: disable-msg=W0613,C6102,C6104,C6105,C6108,C6409


def t_IGNORE_COMMENT(t):
  r'\#.*'
  pass


def t_DQUOTEDSTRING(t):
  r'"[^"]*?"'
  t.lexer.lineno += str(t.value).count('\n')
  return t


def t_newline(t):
  r'\n+'
  t.lexer.lineno += len(t.value)


def t_error(t):
  print "Illegal character '%s' on line %s" % (t.value[0], t.lineno)
  t.lexer.skip(1)


def t_INTEGER(t):
  r'\d+'
  return t


def t_STRING(t):
  r'\w+([-_+.@/]\w*)*'
  # we have an identifier; let's check if it's a keyword or just a string.
  t.type = reserved.get(t.value, 'STRING')
  return t


###
## parser starts here
###
def p_target(p):
  """ target : target header terms
             | """
  if len(p) > 1:
    if type(p[1]) is Policy:
      p[1].AddFilter(p[2], p[3])
      p[0] = p[1]
    else:
      p[0] = Policy(p[2], p[3])


def p_header(p):
  """ header : HEADER '{' header_spec '}' """
  p[0] = p[3]


def p_header_spec(p):
  """ header_spec : header_spec target_spec
                  | header_spec comment_spec
                  | """
  if len(p) > 1:
    if type(p[1]) == Header:
      p[1].AddObject(p[2])
      p[0] = p[1]
    else:
      p[0] = Header()
      p[0].AddObject(p[2])


# we may want to change this at some point if we want to be clever with things
# like being able to set a default input/output policy for iptables policies.
def p_target_spec(p):
  """ target_spec : TARGET ':' ':' strings_or_ints """
  p[0] = Target(p[4])


def p_terms(p):
  """ terms : terms TERM STRING '{' term_spec '}'
            | """
  if len(p) > 1:
    p[5].name = p[3]
    if type(p[1]) == list:
      p[1].append(p[5])
      p[0] = p[1]
    else:
      p[0] = [p[5]]


def p_term_spec(p):
  """ term_spec : term_spec action_spec   # OK
                | term_spec addr_spec
                | term_spec comment_spec
                | term_spec counter_spec
                | term_spec ether_type_spec
                | term_spec exclude_spec
                | term_spec expiration_spec
                | term_spec fragment_offset_spec
                | term_spec icmp_type_spec
                | term_spec interface_spec
                | term_spec logging_spec
                | term_spec losspriority_spec
                | term_spec option_spec
                | term_spec owner_spec
                | term_spec packet_length_spec
                | term_spec platform_spec
                | term_spec policer_spec
                | term_spec port_spec
                | term_spec precedence_spec
                | term_spec principals_spec
                | term_spec prefix_list_spec
                | term_spec protocol_spec
                | term_spec qos_spec
                | term_spec routinginstance_spec
                | term_spec tag_list_spec
                | term_spec timeout_spec
                | term_spec traffic_type_spec
                | term_spec verbatim_spec
                | """
  if len(p) > 1:
    if type(p[1]) == Term:
      p[1].AddObject(p[2])
      p[0] = p[1]
    else:
      p[0] = Term(p[2])


def p_routinginstance_spec(p):
  """ routinginstance_spec : ROUTING_INSTANCE ':' ':' STRING """
  p[0] = VarType(VarType.ROUTING_INSTANCE, p[4])


def p_losspriority_spec(p):
  """ losspriority_spec :  LOSS_PRIORITY ':' ':' STRING """
  p[0] = VarType(VarType.LOSS_PRIORITY, p[4])


def p_precedence_spec(p):
  """ precedence_spec : PRECEDENCE ':' ':' one_or_more_ints """
  p[0] = VarType(VarType.PRECEDENCE, p[4])


def p_icmp_type_spec(p):
  """ icmp_type_spec : ICMP_TYPE ':' ':' one_or_more_strings """
  p[0] = VarType(VarType.ICMP_TYPE, p[4])


def p_packet_length_spec(p):
  """ packet_length_spec : PACKET_LEN ':' ':' INTEGER
                         | PACKET_LEN ':' ':' INTEGER '-' INTEGER """
  if len(p) == 4:
    p[0] = VarType(VarType.PACKET_LEN, str(p[4]))
  else:
    p[0] = VarType(VarType.PACKET_LEN, str(p[4]) + '-' + str(p[6]))


def p_fragment_offset_spec(p):
  """ fragment_offset_spec : FRAGMENT_OFFSET ':' ':' INTEGER
                           | FRAGMENT_OFFSET ':' ':' INTEGER '-' INTEGER """
  if len(p) == 4:
    p[0] = VarType(VarType.FRAGMENT_OFFSET, str(p[4]))
  else:
    p[0] = VarType(VarType.FRAGMENT_OFFSET, str(p[4]) + '-' + str(p[6]))


def p_exclude_spec(p):
  """ exclude_spec : SADDREXCLUDE ':' ':' one_or_more_strings
                   | DADDREXCLUDE ':' ':' one_or_more_strings
                   | ADDREXCLUDE ':' ':' one_or_more_strings
                   | PROTOCOL_EXCEPT ':' ':' one_or_more_strings """

  p[0] = []
  for ex in p[4]:
    if p[1].find('source-exclude') >= 0:
      p[0].append(VarType(VarType.SADDREXCLUDE, ex))
    elif p[1].find('destination-exclude') >= 0:
      p[0].append(VarType(VarType.DADDREXCLUDE, ex))
    elif p[1].find('address-exclude') >= 0:
      p[0].append(VarType(VarType.ADDREXCLUDE, ex))
    elif p[1].find('protocol-except') >= 0:
      p[0].append(VarType(VarType.PROTOCOL_EXCEPT, ex))


def p_prefix_list_spec(p):
  """ prefix_list_spec : DPFX ':' ':' one_or_more_strings
                       | SPFX ':' ':' one_or_more_strings """
  p[0] = []
  for pfx in p[4]:
    if p[1].find('source-prefix') >= 0:
      p[0].append(VarType(VarType.SPFX, pfx))
    elif p[1].find('destination-prefix') >= 0:
      p[0].append(VarType(VarType.DPFX, pfx))


def p_addr_spec(p):
  """ addr_spec : SADDR ':' ':' one_or_more_strings
                | DADDR ':' ':' one_or_more_strings
                | ADDR  ':' ':' one_or_more_strings """
  p[0] = []
  for addr in p[4]:
    if p[1].find('source-address') >= 0:
      p[0].append(VarType(VarType.SADDRESS, addr))
    elif p[1].find('destination-address') >= 0:
      p[0].append(VarType(VarType.DADDRESS, addr))
    else:
      p[0].append(VarType(VarType.ADDRESS, addr))


def p_port_spec(p):
  """ port_spec : SPORT ':' ':' one_or_more_strings
                | DPORT ':' ':' one_or_more_strings
                | PORT ':' ':' one_or_more_strings """
  p[0] = []
  for port in p[4]:
    if p[1].find('source-port') >= 0:
      p[0].append(VarType(VarType.SPORT, port))
    elif p[1].find('destination-port') >= 0:
      p[0].append(VarType(VarType.DPORT, port))
    else:
      p[0].append(VarType(VarType.PORT, port))


def p_protocol_spec(p):
  """ protocol_spec : PROTOCOL ':' ':' strings_or_ints """
  p[0] = []
  for proto in p[4]:
    p[0].append(VarType(VarType.PROTOCOL, proto))


def p_tag_list_spec(p):
  """ tag_list_spec : DTAG ':' ':' one_or_more_strings
                    | STAG ':' ':' one_or_more_strings """
  p[0] = []
  for tag in p[4]:
    if p[1].find('source-tag') >= 0:
      p[0].append(VarType(VarType.STAG, tag))
    elif p[1].find('destination-tag') >= 0:
      p[0].append(VarType(VarType.DTAG, tag))


def p_ether_type_spec(p):
  """ ether_type_spec : ETHER_TYPE ':' ':' one_or_more_strings """
  p[0] = []
  for proto in p[4]:
    p[0].append(VarType(VarType.ETHER_TYPE, proto))


def p_traffic_type_spec(p):
  """ traffic_type_spec : TRAFFIC_TYPE ':' ':' one_or_more_strings """
  p[0] = []
  for proto in p[4]:
    p[0].append(VarType(VarType.TRAFFIC_TYPE, proto))


def p_policer_spec(p):
  """ policer_spec : POLICER ':' ':' STRING """
  p[0] = VarType(VarType.POLICER, p[4])


def p_logging_spec(p):
  """ logging_spec : LOGGING ':' ':' STRING """
  p[0] = VarType(VarType.LOGGING, p[4])


def p_option_spec(p):
  """ option_spec : OPTION ':' ':' one_or_more_strings """
  p[0] = []
  for opt in p[4]:
    p[0].append(VarType(VarType.OPTION, opt))

def p_principals_spec(p):
  """ principals_spec : PRINCIPALS ':' ':' one_or_more_strings """
  p[0] = []
  for opt in p[4]:
    p[0].append(VarType(VarType.PRINCIPALS, opt))

def p_action_spec(p):
  """ action_spec : ACTION ':' ':' STRING """
  p[0] = VarType(VarType.ACTION, p[4])


def p_counter_spec(p):
  """ counter_spec : COUNTER ':' ':' STRING """
  p[0] = VarType(VarType.COUNTER, p[4])


def p_expiration_spec(p):
  """ expiration_spec : EXPIRATION ':' ':' INTEGER '-' INTEGER '-' INTEGER """
  p[0] = VarType(VarType.EXPIRATION, datetime.date(int(p[4]),
                                                   int(p[6]),
                                                   int(p[8])))


def p_comment_spec(p):
  """ comment_spec : COMMENT ':' ':' DQUOTEDSTRING """
  p[0] = VarType(VarType.COMMENT, p[4])


def p_owner_spec(p):
  """ owner_spec : OWNER ':' ':' STRING """
  p[0] = VarType(VarType.OWNER, p[4])


def p_verbatim_spec(p):
  """ verbatim_spec : VERBATIM ':' ':' STRING DQUOTEDSTRING """
  p[0] = VarType(VarType.VERBATIM, [p[4], p[5].strip('"')])


def p_qos_spec(p):
  """ qos_spec : QOS ':' ':' STRING """
  p[0] = VarType(VarType.QOS, p[4])


def p_interface_spec(p):
  """ interface_spec : SINTERFACE ':' ':' STRING
                     | DINTERFACE ':' ':' STRING """
  if p[1].find('source-interface') >= 0:
    p[0] = VarType(VarType.SINTERFACE, p[4])
  elif p[1].find('destination-interface') >= 0:
    p[0] = VarType(VarType.DINTERFACE, p[4])


def p_platform_spec(p):
  """ platform_spec : PLATFORM ':' ':' one_or_more_strings
                    | PLATFORMEXCLUDE ':' ':' one_or_more_strings """
  p[0] = []
  for platform in p[4]:
    if p[1].find('platform-exclude') >= 0:
      p[0].append(VarType(VarType.PLATFORMEXCLUDE, platform))
    elif p[1].find('platform') >= 0:
      p[0].append(VarType(VarType.PLATFORM, platform))


def p_timeout_spec(p):
  """ timeout_spec : TIMEOUT ':' ':' INTEGER """
  p[0] = VarType(VarType.TIMEOUT, p[4])


def p_one_or_more_strings(p):
  """ one_or_more_strings : one_or_more_strings STRING
                          | STRING
                          | """
  if len(p) > 1:
    if type(p[1]) == type([]):
      p[1].append(p[2])
      p[0] = p[1]
    else:
      p[0] = [p[1]]


def p_one_or_more_ints(p):
  """ one_or_more_ints : one_or_more_ints INTEGER
                      | INTEGER
                      | """
  if len(p) > 1:
    if type(p[1]) == type([]):
      p[1].append(p[2])
      p[0] = p[1]
    else:
      p[0] = [p[1]]


def p_strings_or_ints(p):
  """ strings_or_ints : strings_or_ints STRING
                      | strings_or_ints INTEGER
                      | STRING
                      | INTEGER
                      | """
  if len(p) > 1:
    if type(p[1]) is list:
      p[1].append(p[2])
      p[0] = p[1]
    else:
      p[0] = [p[1]]


def p_error(p):
  """."""
  next_token = yacc.token()
  if next_token is None:
    use_token = 'EOF'
  else:
    use_token = repr(next_token.value)

  if p:
    raise ParseError(' ERROR on "%s" (type %s, line %d, Next %s)'
                     % (p.value, p.type, p.lineno, use_token))
  else:
    raise ParseError(' ERROR you likely have unablanaced "{"\'s')

# pylint: enable-msg=W0613,C6102,C6104,C6105,C6108,C6409


def memoize(obj):
  """Memoize decorator for objects that take args and or kwargs."""

  cache = obj.cache = {}

  @wraps(obj)
  def memoizer(*args, **kwargs):
    key = (args, tuple(zip(kwargs.iteritems())))
    try:
      return cache[key]
    except KeyError:
      value = obj(*args, **kwargs)
      cache[key] = value
      return value
  return memoizer


def _ReadFile(filename):
  """Read data from a file if it exists.

  Args:
    filename: str - Filename

  Returns:
    data: str contents of file.

  Raises:
    FileNotFoundError: if requested file does not exist.
    FileReadError: Any error resulting from trying to open/read file.
  """
  if os.path.exists(filename):
    try:
      data = open(filename, 'r').read()
      return data
    except IOError:
      raise FileReadError('Unable to open or read file %s' % filename)
  else:
    raise FileNotFoundError('Unable to open policy file %s' % filename)


def _Preprocess(data, max_depth=5, base_dir=''):
  """Search input for include statements and import specified include file.

  Search input for include statements and if found, import specified file
  and recursively search included data for includes as well up to max_depth.

  Args:
    data: A string of Policy file data.
    max_depth: Maximum depth of included files
    base_dir: Base path string where to look for policy or include files

  Returns:
    A string containing result of the processed input data

  Raises:
    RecursionTooDeepError: nested include files exceed maximum
  """
  if not max_depth:
    raise RecursionTooDeepError('%s' % (
        'Included files exceed maximum recursion depth of %s.' % max_depth))
  rval = []
  lines = [x.rstrip() for x in data.splitlines()]
  for index, line in enumerate(lines):
    words = line.split()
    if len(words) > 1 and words[0] == '#include':
      # remove any quotes around included filename
      include_file = words[1].strip('\'"')
      data = _ReadFile(os.path.join(base_dir, include_file))
      # recursively handle includes in included data
      inc_data = _Preprocess(data, max_depth - 1, base_dir=base_dir)
      rval.extend(inc_data)
    else:
      rval.append(line)
  return rval


def ParseFile(filename, definitions=None, optimize=True, base_dir='',
              shade_check=False):
  """Parse the policy contained in file, optionally provide a naming object.

  Read specified policy file and parse into a policy object.

  Args:
    filename: Name of policy file to parse.
    definitions: optional naming library definitions object.
    optimize: bool - whether to summarize networks and services.
    base_dir: base path string to look for acls or include files.
    shade_check: bool - whether to raise an exception when a term is shaded.

  Returns:
    policy object.
  """
  data = _ReadFile(filename)
  p = ParsePolicy(data, definitions, optimize, base_dir=base_dir,
                  shade_check=shade_check)
  return p


@memoize
def CacheParseFile(*args, **kwargs):
  """Same as ParseFile, but cached if possible.

  If this was previously called with same args/kwargs, then just return
  the previous result from cache.

  See the ParseFile function for signature details.
  """

  return ParseFile(*args, **kwargs)


def ParsePolicy(data, definitions=None, optimize=True, base_dir='',
                shade_check=False):
  """Parse the policy in 'data', optionally provide a naming object.

  Parse a blob of policy text into a policy object.

  Args:
    data: a string blob of policy data to parse.
    definitions: optional naming library definitions object.
    optimize: bool - whether to summarize networks and services.
    base_dir: base path string to look for policies or include files.
    shade_check: bool - whether to raise an exception when a term is shaded.

  Returns:
    policy object.
  """
  try:
    if definitions:
      globals()['DEFINITIONS'] = definitions
    else:
      globals()['DEFINITIONS'] = naming.Naming(DEFAULT_DEFINITIONS)
    if not optimize:
      globals()['_OPTIMIZE'] = False
    if shade_check:
      globals()['_SHADE_CHECK'] = True

    lexer = lex.lex()

    preprocessed_data = '\n'.join(_Preprocess(data, base_dir=base_dir))
    p = yacc.yacc(write_tables=False, debug=0, errorlog=yacc.NullLogger())

    return p.parse(preprocessed_data, lexer=lexer)

  except IndexError:
    return False


# If you call this from the command line, you can specify a policy file for it
# to read.
if __name__ == '__main__':
  ret = 0
  if len(sys.argv) > 1:
    try:
      ret = ParsePolicy(open(sys.argv[1], 'r').read())
    except IOError:
      print('ERROR: \'%s\' either does not exist or is not readable' %
            (sys.argv[1]))
      ret = 1
  else:
    # default to reading stdin
    ret = ParsePolicy(sys.stdin.read())
  sys.exit(ret)
