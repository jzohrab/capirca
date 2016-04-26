# Build a parser
parse_acl_base.py: acl.ebnf
	PYTHONPATH=.. python -m grako -m ACL -o parse_acl_base.py acl.ebnf 2>&1

# Spike parse it (run)
r: parse_acl_base.py
	python spike_grako.py
