#!/bin/python

verbose=False

#   *----------------...-*
#   |  P_RULE GENERATOR  |
#   | ------------------ |
#   |  yoinked from Tim  |
#   |  and modified      |
#   *--------------------*

def generator():
	count = {}
	def gen(name):
		nonlocal count
		if name not in count: count[name] = 0
		else: count[name] += 1
		return name + str(count[name])
	return gen

unique = generator()
module = __import__(__name__)

def rule_func(name, rule, func):
	def f(p):
		if verbose: print(rule, '<-', *p)
		p[0] = func(p)
	f.__doc__ = rule
	setattr(module, unique(f'p_{name}'), f)

def rule_node(tag, rule, *children):
	def f(p):
		c = (p[child] if isinstance(child, int) else child for child in children)
		return (tag, *c)
	rule_func(tag, rule, f)

def rule_op(op, fix='infix', prec=None, func=None):
	prec = f'%prec {prec}' if prec else ''
	match(fix):
		case 'infix' : rule_func('binop', f"exp : exp {op} exp {prec}", lambda p: ('op', p[2], p[1], p[3]))
		case 'prefix': rule_func('preop', f"exp : {op} exp {prec}", lambda p: ('op', p[1], p[2]))
		case 'suffix': rule_func('sufop', f"exp : exp {op} {prec}", lambda p: ('op', p[2], p[1]))
		case _: raise Exception("that ain't fixing nothing")

def rule_list(name, elem, sep, trailing_seperator='disallow'):
	rule_func(name, f"{name} : {elem} {sep} {name}", lambda p: [p[1], *p[3]])
	if trailing_seperator != 'dissallow':
		rule_func(name, f"{name} : {elem} {sep}", lambda p: [p[1]])
	if trailing_seperator != 'force':
		rule_func(name, f"{name} : {elem}", lambda p: [p[1]])

#   *-----...-*
#   |  LEXER  |
#   *---------*

from ply.lex import lex

literals = r'+-*(){};[],'
reserved = {'mod': 'DIV', 'imag':'imag', 'e':'POW', 'or': 'or', 'xor': 'xor', 'and': 'and', 'not':'not',
		'echo': 'SYS', 'load': 'SYS',
		'solange': 'WHILE', 'für': 'FOR', 'in': 'IN',
		'⟳' :'WHILE', '∀': 'FOR', '∈': 'IN'}
tokens = ['NUM', 'STR', 'ID', 'ASG', 'USG', 'DIV', 'POW', 'CMP'] + list(reserved.values()) + ['IF', 'THEN', 'DO', 'ELSE', 'END']
tokens = list(set(tokens))

def t_IF(t):
	r'wenn|¿'
	return t

def t_DO(t):
	r',\s*mach|:'
	return t

def t_THEN(t):
	r'gilt|\?'
	return t

def t_ELSE(t):
	r',\s*sonst|\!'
	return t

t_END = r'\.'

def t_ID(t):
	r'[a-zA-Z\u00a0-\U0001f645_][a-zA-Z\u00a0-\U0001f645_0-9]*'
	t.type = reserved.get(t.value.lower(), 'ID')
	return t

t_NUM = r'(0|[1-9][0-9]*)\.([0-9]*[1-9]|0)' + '|' + r'0b0|0b1[0|1]*|0x0|0x[1-9a-fA-F][0-9a-fA-F]*|0|[1-9][0-9]*'
t_STR = r'".*"'
t_ASG = r'='
t_USG = r'(\+\+|--)((?!' + t_NUM + '|' + t_ID.__doc__ + ')|(?=imag))' # edge cases: ['1++1', '1++a', 'a++imag']
t_DIV = r'[|\/\\]'
t_POW = r'\*\*'
t_CMP = r'[!=<>]=|[<>]'

t_ignore_CMT = r'\#[^#]*\#'
t_ignore = ' \t'

def t_newline(t):
	r'\n+'
	t.lexer.lineno += len(t.value)

def t_error(t):
	raise SyntaxError(f'illegal token {t.value}')

lexer = lex()

#   *------...-*
#   |  PARSER  |
#   *----------*

from ply.yacc import yacc

precedence = [	['left', 'ID'],
		['right', 'ASG', 'SYS', 'IF'],
		['left', 'or'], ['left', 'xor'], ['left', 'and'],
		['left', 'CLS'], ['left', 'CMP'],
		['left', '+', '-'],
		['left', '*', 'DIV'],
		['right', 'POW'],
		['left', 'not', 'imag'],
		['right', 'USG'],
]

# simples
rule_node('id',  'exp : ID', 1)
rule_node('val', 'exp : NUM', 1)
rule_node('str', 'exp : STR', 1)

# operators
for binop in ["'+'", "'-'", "'*'", 'DIV', 'POW', 'or', 'xor', 'and']:
	rule_op(binop)
	rule_func('op', f'exp : ID {binop} exp', lambda p: ('op', p[2], ('id', p[1]), p[3]))
	rule_func('asg', f'exp : ID {binop} ASG exp', lambda p: ('asg', p[3], p[1], ('op', p[2], ('id', p[1]), p[4])))
for unpre in ["'+'", "'-'"]:
	rule_func('op', f'exp : {unpre} exp', lambda p: ('op', 'u'+p[1], p[2]))
rule_op('not',  fix='prefix')
rule_op('imag', fix='suffix')

# groups
rule_func('grp', r"exp : '(' exp ')'", lambda p: p[2])

# sequences
rule_list('seq', 'exp', "';'", trailing_seperator='')
rule_func('seq', "exp : '{' seq '}'", lambda p: ('seq', *p[2]))

# assign, increment & decrement
rule_node('asg', 'exp : ID ASG exp', 2, 1, 3)
rule_node('asg', 'exp : ID USG', 2, 1)

# comperator lists
# only work if compare operators all have same precedence (idky)
rule_func('cmp', 'cmp : exp CMP exp', lambda p: [[p[2]], [p[1], p[3]]])
rule_func('cmp', 'cmp : cmp CMP exp', lambda p: [p[1][0] + [p[2]], p[1][1] + [p[3]]])
rule_func('cmp', 'exp : cmp %prec CLS', lambda p: ('cmp', *p[1]))

# control structures
rule_func('ctl', 'ifc : IF exp THEN exp', lambda p: ('if', p[2], p[4], None))
rule_func('ctl', 'ifc : IF exp THEN exp ELSE exp', lambda p: ('if', p[2], p[4], p[6]))
rule_func('ctl', 'ifc : IF exp THEN exp ELSE ifc %prec IF', lambda p: ('if', p[2], p[4], p[6]))
rule_func('ctl', 'exp : ifc END', lambda p: p[1])

# iterator
rule_func('it', "it : '[' exp ',' exp ']'", lambda p: ('it', p[2], p[4], 1))
rule_func('it', "it : ']' exp ',' exp ']'", lambda p: ('it', p[2], p[4], 2))
rule_func('it', "it : '[' exp ',' exp '['", lambda p: ('it', p[2], p[4], 3))
rule_func('it', "it : ']' exp ',' exp '['", lambda p: ('it', p[2], p[4], 4))
rule_func('it', 'exp : it', lambda p: p[1])

# loops
rule_func('for', 'exp : FOR ID IN it DO exp END', lambda p: ('for', p[2], p[4], p[6]))
rule_func('while', 'exp : WHILE exp DO exp END', lambda p: ('while', p[2], p[4]))

# load and echo
rule_op('SYS', fix='prefix')

def p_error(p):
	raise SyntaxError(f'Syntax error in {p}')

parser = yacc(start='exp')

#   *-----------...-*
#   |  INTERPRETER  |
#   *---------------*

NONE=float('NaN')

import math, cmath

def int2str(x):
	try:
		return bytes.fromhex(hex(x)[2:].zfill(2)).decode('ascii')
	except Exception: return '\uFFFD'
		

ops = { '+'	: lambda x,y: x+y,
	'-'	: lambda x,y: x-y,
	'*'	: lambda x,y: x*y,
	'|'	: lambda x,y: x/y,
	'/'	: lambda x,y: math.ceil(x/y),
	'\\'	: lambda x,y: math.floor(x/y),
	'mod'	: lambda x,y: x % y,
	'**'	: lambda x,y: x**y,
	'e'	: lambda x,y: x * 10**y,
	'imag'	: lambda x: x * 1j,
	'u+'	: lambda x: abs(x),
	'u-'	: lambda x: -x,
	'or'	: lambda x,y: int(bool(x) or  bool(y)),
	'xor'	: lambda x,y: int(bool(x) ==  bool(y)),
	'and'	: lambda x,y: int(bool(x) and bool(y)),
	'not'	: lambda x: int(not bool(x)),
	'echo'	: lambda x: print('\x1b[0;33m' + int2str(x) + '\x1b[0m') or x,
	'load'	: lambda x: eval(parser.parse(open(int2str(x), 'r').read()), env) or NONE}

cmp = { '<'	: lambda x,y: x <  y,
	'<='	: lambda x,y: x <= y,
	'=='	: lambda x,y: x == y,
	'!='	: lambda x,y: x != y,
	'>='	: lambda x,y: x >= y,
	'>'	: lambda x,y: x >  y,}

pyeval = eval
def eval(exp, env):
	match(exp):
		case ('op', op, *args):
			return ops[op](*[eval(x, env) for x in args])
		case ('cmp', op, x):
			x = [eval(xi, env) for xi in x]
			return int(all([cmp[op[i]](x[i], x[i+1]) for i in range(len(op))]))
		case ('val', x):
			return pyeval(x) # zieh, zieh, zieh
		case ('str', x):
			return pyeval('0x' + pyeval(x).encode().hex())
		case ('asg', op, x, *exp):
			if x not in env: env[x] = 0
			if op == '=':  env[x] = eval(*exp, env)
			if op == '++': env[x]  += 1
			if op == '--': env[x]  -= 1
			return env[x]
		case ('id', x):
			if x not in env: env[x] = 0
			return env[x]
		case ('seq', *exp, ret):
			for e in exp:
				eval(e, env)
			return eval(ret, env)
		case ('if', con, exp, alt):
			if eval(con, env): return eval(exp, env)
			return eval(alt, env) if alt else NONE
		case ('it', lo, up, t):
			return Iterator(eval(lo, env), eval(up, env), t)
		case ('for', i, it, exp):
			it = eval(it, env)
			ret = NONE
			while not it.empty():
				env[i] = it.next()
				ret = eval(exp, env)
			return ret
		case ('while', cond, exp):
			ret = NONE
			while eval(cond, env):
				ret = eval(exp, env)
			return ret
		case _: raise Exception(f'exception in {exp}')

class Iterator():
	def __init__(self, lo, up, t, step=1):
		self.lo = lo
		self.up = up
		self.t  = t
		self.step = step

	def __repr__(self):
		match(self.t):
			case 1: return f'[{self.lo},{self.up}]'
			case 2: return f']{self.lo},{self.up}]'
			case 3: return f'[{self.lo},{self.up}['
			case 4: return f']{self.lo},{self.up}['

	def contains(self, x):
		match(self.t):
			case 1: return self.lo <  x <  self.up
			case 2: return self.lo <= x <  self.up
			case 3: return self.lo <  x <= self.up
			case 4: return self.lo <= x <= self.up

	def empty(self):
		if self.t < 3:
			return self.lo + self.step >= self.up
		else:   return self.lo + self.step > self.up

	def next(self):
		if self.empty(): return NONE
		self.lo += self.step
		if self.t % 2 == 1:
			return self.lo
		else:   return self.lo - self.step

#   *----...-*
#   |  MAIN  |
#   *--------*

if __name__ == '__main__':
	import os, readline, argparse, sys

	argp = argparse.ArgumentParser(prog=sys.argv[0], description='ICC25 Interpreter\nInterpretation und Compilation von Computerprogrammen 2025\nHochschule Bonn-Rhein-Sieg',
	epilog='󱤹 Raphael Schönefeld', formatter_class=argparse.RawTextHelpFormatter)
	argp.add_argument('-i', '--input', help='ICC25 source code')
	argp.add_argument('-v', '--verbose', action='store_true')
	args = argp.parse_args()

	verbose = args.verbose
	if args.input:
		with open(args.input, 'r') as file:
			src = file.read()[:-1]
			res = eval(parser.parse(src), {})
			print(res)
			sys.exit(0)

	hist = os.path.join(os.path.expanduser("~"), ".icc_history")
	try:
		readline.read_history_file(hist)
		readline.set_history_length(1000)
	except FileNotFoundError:
		pass

	def exit():
		print()
		if not os.path.exists(hist): os.mknod(hist)
		readline.write_history_file(hist)
		sys.exit(0)

	env = {}
	while True:
		try:
			src = input('\x1b[0;32m' + '>>> ' + '\x1b[0m')
			while src.count('#') % 2 != 0 or src.count('{') != src.count('}'):
				src += '\n' + input('\x1b[0;32m' + '... ' + '\x1b[0m')
			result = parser.parse(src)
			if verbose: print(result)
			if result: print(eval(result, env))
		except EOFError: exit()
		except Exception as e: print('\x1b[0;31m' + repr(e) + '\x1b[0m')
		except KeyboardInterrupt as e: print('\n\x1b[0;31m' + 'KeyboardInterrupt' + '\x1b[0m')
