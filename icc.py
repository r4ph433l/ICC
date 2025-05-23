#!/bin/python
verbose=False
write_tables=False

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

def rule_op(op, fix='infix', prec=None, use_val=True):
    prec = f'%prec {prec}' if prec else ''
    match(fix):
        case 'infix' : rule_func('binop', f"exp : exp {op} exp {prec}", lambda p: ('op', p[2] if use_val else op, p[1], p[3]))
        case 'prefix': rule_func('preop', f"exp : {op} exp {prec}", lambda p: ('op', p[1] if use_val else op, p[2]))
        case 'suffix': rule_func('sufop', f"exp : exp {op} {prec}", lambda p: ('op', p[2] if use_val else op, p[1]))
        case _: raise Exception("that ain't fixing nothing")

def rule_list(name, elem, sep, prec=None, trailing_seperator='disallow'):
    prec = f'%prec {prec}' if prec else ''
    rule_func(name, f"{name} : {elem} {sep} {name} {prec}", lambda p: [p[1], *p[3]])
    if trailing_seperator != 'dissallow':
        rule_func(name, f"{name} : {elem} {sep} {prec}", lambda p: [p[1]])
    if trailing_seperator != 'force':
        rule_func(name, f"{name} : {elem} {prec}", lambda p: [p[1]])

def _t(name, reg):
    def f(p):
        return p
    f.__doc__ = reg
    setattr(module, f't_{name}', f)

#   *-----...-*
#   |  LEXER  |
#   *---------*

from ply.lex import lex

literals = r'+-*(){};[],:'
restoken = ['MOD', 'OR', 'XOR', 'AND', 'NOT', 'IF', 'IN', 'ELSE', 'FOR', 'WHILE', 'ECHO', 'READ', 'LOAD', 'EVAL', 'SIZE']
engwords = {s.lower(): s for s in restoken}
reserved = engwords.copy()
tokens = ['NUM', 'STR', 'ID', 'STK', 'ASG', 'USG', 'DIV', 'POW', 'CMP', 'END', 'TO'] + restoken

_t('MOD', '≡')
_t('OR', '∨'); _t('XOR', '⊻'); _t('AND', '∧'); _t('NOT', '¬')
_t('IF', r'\?'); _t('ELSE', r'\!'); _t('FOR', '∀'); _t('IN', '∈'); _t('WHILE', '⟲')
_t('ECHO', '♫'); _t('READ', '📖'); _t('LOAD', '⊃'); _t('EVAL', '⊢'); _t('SIZE', '‖')

def t_ID(t):
    r'[a-zA-Z\u00a0-\U0001f645_][a-zA-Z\u00a0-\U0001f645_0-9]*'
    t.type = reserved.get(t.value, 'ID')
    return t

t_NUM = r'((0|[1-9][0-9]*)\.([0-9]*[1-9]|0)|0b0|0b1[0|1]*|0x0|0x[1-9a-fA-F][0-9a-fA-F]*|0|[1-9][0-9]*)j?'
t_STR = r'"[^"]*"'
t_ASG = r'='
t_USG = r'(\+\+|--)((?!' + t_NUM + '|' + t_ID.__doc__ + ')|(?=imag))' # edge cases: ['1++1', '1++a', 'a++imag']
t_DIV = r'[|\/\\]'
t_POW = r'\*\*'
t_CMP = r'[!=<>]=|[<>]'
t_END = r'\.'
t_STK = r'\$'
t_TO  = r'->'

t_ignore = ' \t'
def t_comment(t):
    r'\#[^#]*\#'
    t.lexer.lineno += t.value.count(r'\n')

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

precedence = [
        ['left', 'ID'],
        ['right', 'ASG', 'SYS', 'IF'],
        ['right', 'TO'],
        ['left', 'ARR'],['left', ','],
        ['left', 'OR'], ['left', 'XOR'], ['left', 'AND'],
        ['left', 'CLS'], ['left', 'CMP'],
        ['left', '+', '-'],
        ['left', '*', 'DIV', 'MOD'],
        ['right', 'POW'],
        ['left', 'NOT', 'STK'],
        ['right', 'USG'],
        ['left', '['],
]

# simples
rule_node('id',  'exp : ID',  1)
rule_node('val', 'exp : NUM', 1)
rule_node('val', 'exp : STR', 1)

# operators
for binop in ["'+'", "'-'", "'*'", 'DIV']:
    rule_op(binop)
    rule_func('op', f'exp : ID {binop} exp', lambda p: ('op', p[2], ('id', p[1]), p[3]))
    rule_func('asg', f'exp : ID {binop} ASG exp', lambda p: ('asg', p[3], p[1], ('op', p[2], ('id', p[1]), p[4])))
for binop in ['MOD', 'POW', 'OR', 'XOR', 'AND']:
    rule_op(binop, use_val=False)
    rule_func('op', f'exp : ID {binop} exp', lambda p: ('op', binop, ('id', p[1]), p[3]))
    rule_func('asg', f'exp : ID {binop} ASG exp', lambda p: ('asg', p[3], p[1], ('op', binop, ('id', p[1]), p[4])))

for unpre in ["'+'", "'-'"]:
    rule_func('op', f'exp : {unpre} exp', lambda p: ('op', 'u'+p[1], p[2]))
rule_op('NOT',  fix='prefix')

# groups
rule_func('grp', r"exp : '(' exp ')'", lambda p: p[2])

# sequences
rule_list('seq', 'exp', "';'", trailing_seperator='')
rule_func('seq', "exp : '{' seq '}'", lambda p: ('seq', *p[2]))

# assign, increment & decrement
rule_node('asg', 'exp : ID ASG exp', 2, 1, 3)
rule_node('asg', 'exp : ID USG', 2, 1)
rule_func('asg', "exp : agt ASG exp", lambda p: ('a_asg', p[2], p[1], p[3]))
rule_func('asg', "exp : agt USG", lambda p: ('a_asg', p[2], p[1]))

# comperator lists
# only work if compare operators all have same precedence (idky)
rule_func('cmp', 'cmp : exp CMP exp', lambda p: [[p[2]], [p[1], p[3]]])
rule_func('cmp', 'cmp : cmp CMP exp', lambda p: [p[1][0] + [p[2]], p[1][1] + [p[3]]])
rule_func('cmp', 'exp : cmp %prec CLS', lambda p: ('cmp', *p[1]))

# arrays
rule_func('arr', "arr : exp ',' exp", lambda p: [p[1], p[3]])
rule_func('arr', "arr : arr ',' exp", lambda p: [*p[1], p[3]])
rule_func('arr', "exp : arr %prec ARR", lambda p: ('arr', *p[1]))
rule_func('arr', "exp : '(' ')'", lambda p: ('arr',))
rule_func('arr', "exp : '(' exp ',' ')'", lambda p: ('arr', p[2]))
rule_func('arr', "agt : exp '[' exp ']'", lambda p: ('a_get', p[1], p[3]))
rule_func('arr', 'exp : agt', lambda p: p[1])

# lambda, functions and calls
rule_func('fun', 'exp : exp TO exp', lambda p: ('lambda', p[1], p[3], False))
rule_func('fun', "exp : exp STK TO exp", lambda p: ('lambda', p[1], p[4], True))
rule_func('fun', "exp : exp '(' exp ')'", lambda p: ('call', p[1], p[3]))
rule_func('fun', "exp : exp '(' ')'", lambda p: ('call', p[1], ('arr',)))
rule_func('stk', "exp : STK exp", lambda p: ('stk', p[2]))

# control structures
rule_func('ctl', "ifc : IF exp ':' exp", lambda p: ('if', p[2], p[4], None))
rule_func('ctl', "ifc : IF exp ':' exp ELSE exp", lambda p: ('if', p[2], p[4], p[6]))
rule_func('ctl', "ifc : IF exp ':' exp ELSE ifc %prec IF", lambda p: ('if', p[2], p[4], p[6]))
rule_func('ctl', 'exp : ifc END', lambda p: p[1])

# iterator
rule_func('it', "it : '[' exp ',' exp ']'", lambda p: ('range', p[2], p[4], (True,  True )))
rule_func('it', "it : ']' exp ',' exp ']'", lambda p: ('range', p[2], p[4], (False, True )))
rule_func('it', "it : '[' exp ',' exp '['", lambda p: ('range', p[2], p[4], (True,  False)))
rule_func('it', "it : ']' exp ',' exp '['", lambda p: ('range', p[2], p[4], (False, False)))
rule_func('it', "exp : it", lambda p: p[1])
rule_func('it', 'exp : exp IN exp %prec CMP', lambda p: ('in', p[1], p[3]))

# loops
rule_func('for', "exp : FOR ID IN exp ':' exp END", lambda p: ('for', p[2], p[4] ,p[6]))
rule_func('for', "exp : FOR NUM ID IN it ':' exp END", lambda p: ('for', p[3], p[5], p[7], ('val', p[2])))
rule_func('while', "exp : WHILE exp ':' exp END", lambda p: ('while', p[2], p[4]))

# print and echo
for sys in ['ECHO', 'READ', 'LOAD', 'EVAL', 'SIZE']:
    rule_op(sys, fix='prefix', prec='SYS', use_val=False)

def p_error(p):
    raise SyntaxError(f'Syntax error in {p}')

#   *-----------...-*
#   |  INTERPRETER  |
#   *---------------*

NONE=float('NaN')

import math, cmath

pyeval = eval
ops = { 
    '+' : lambda x,y: x+y,
    '-'     : lambda x,y: x-y,
    '*'     : lambda x,y: x*y,
    '|'     : lambda x,y: x/y,
    '/'     : lambda x,y: math.ceil(x/y),
    '\\'    : lambda x,y: math.floor(x/y),
    'MOD'   : lambda x,y: x % y,
    '**'    : lambda x,y: x**y,
    'u+'    : lambda x: abs(x),
    'u-'    : lambda x: -x,
    'OR'    : lambda x,y: int(bool(x) or  bool(y)),
    'XOR'   : lambda x,y: int(bool(x) ==  bool(y)),
    'AND'   : lambda x,y: int(bool(x) and bool(y)),
    'NOT'   : lambda x: int(not bool(x)),
    'ECHO'  : lambda x: (print(x.format(**env.vars), end='') if isinstance(x, str) else print(x)) or NONE,
    'READ'  : lambda x: eval(parser.parse(input(x.format(**env.vars) if isinstance(x, str) else x)), env),
    'LOAD'  : lambda x: load(x),
    'EVAL'  : lambda x: pyeval(x) or NONE,
    'SIZE'  : lambda x: x.len() if hasattr(x, 'len') else len(x) if hasattr(x, '__len__') else NONE,
}

def load(path):
    global reserved
    tmp, reserved = reserved.copy(), engwords.copy()
    ret = NONE
    with open(path, 'r') as file:
        file = file.read()
        if file.startswith('#?'):
            lanbang, file = file.split('\n', 1)
            language(lanbang[2:])
        ret = eval(parser.parse(file), env) 
    reserved = tmp
    return ret

cmp = { '<'     : lambda x,y: x <  y,
        '<='    : lambda x,y: x <= y,
        '=='    : lambda x,y: x == y,
        '!='    : lambda x,y: x != y,
        '>='    : lambda x,y: x >= y,
        '>'     : lambda x,y: x >  y,
}

def eval(exp, env):
    match(exp):
        case ('op', op, *args):
            return ops[op](*[eval(x, env) for x in args])
        case ('cmp', op, x):
            x = [eval(xi, env) for xi in x]
            return int(all([cmp[op[i]](x[i], x[i+1]) for i in range(len(op))]))
        case ('val', x):
            return pyeval(x) # zieh, zieh, zieh
        case ('id', x):
            return env[x]
        case ('asg', op, x, *exp):
            match(op):
                case '=':  env[x]  = eval(*exp, env)
                case '++': env[x] += 1
                case '--': env[x] -= 1
            return env[x]
        case ('arr', *x):
            arr = []
            dic = {}
            for xi in x:
                if xi[0] == 'asg':
                    dic[xi[2]] = eval(xi, env.fork())
                else: arr.append(eval(xi, env))
            return Array(arr, dic)
        case ('a_get', a, i):
            i = eval(i, env)
            a = eval(a, env)
            return a[i]
        case ('a_asg', op, x, *exp):
            path = []
            # crawl through the ast
            while x[0] == 'a_get':
                _, x, i = x
                path.append(eval(i, env))
            path = path[::-1]
            x = eval(x, env)
            # get last possible lvalue
            for i in path[:-1]:
                x = x[i]
            match(op):
                case '=':  x[path[-1]] = eval(*exp, env)
                case '++': x[path[-1]] += 1
                case '--': x[path[-1]] -= 1
            return x[path[-1]]
        case ('seq', *exp, ret):
            for e in exp:
                eval(e, env)
            return eval(ret, env)
        case ('if', con, exp, alt):
            if eval(con, env): return eval(exp, env)
            return eval(alt, env) if alt else NONE
        case ('range', lo, up, inc):
            return Range(eval(lo, env), eval(up, env), inc)
        case ('in', exp, it):
            return int(eval(exp, env) in eval(it, env))
        case ('for', i, it, exp, *n):
            it = eval(it, env)
            if not hasattr(it, '__iter__'): raise Exception(f'{it} is not iterable')
            if len(n) > 0: it.step *= eval(*n, env)
            ret = NONE
            for x in it:
                env[i] = x
                ret = eval(exp, env)
            return ret
        case ('while', cond, exp):
            ret = NONE
            while eval(cond, env):
                ret = eval(exp, env)
            return ret
        case ('lambda', arg, fun, stk):
            t, *arg = arg
            if t == 'arr':
                for i in range(len(arg)):
                    if arg[i][0] != 'id': raise Exception(f'arg {arg[i]} is illegal')
                    arg[i] = arg[i][1]
            elif t != 'id': raise Exception(f'arg {arg} is illegal')
            return Lambda(env, arg, fun, stk)
        case ('call', fun, arg):
            env_f, arg_f, fun, stk = eval(fun, env)
            env_f = env_f.fork()
            arg_f = arg_f.copy()
            arg = eval(arg, env)
            # arg can be array or exp
            if not isinstance(arg, Array):
                arg = [arg]
            else:
                env_f.vars |= arg.dic
                for a in arg.dic.keys():
                    arg_f.remove(a)
            # match given args to lambda args
            while arg_f and arg:
                env_f[arg_f.pop(0)] = arg.pop(0)
            # undersupply
            if arg_f:
                return Lambda(env_f, arg_f, fun, stk)
            # oversupply
            if arg and not stk:
                raise Exception(f'too many arguments')
            for x in arg:
                env_f.stk.append(x)
            return eval(fun, env_f)
        case ('stk', i):
            i = eval(i, env)
            if i == 0: return len(env.stk)
            return env.stk[i-1]
        case _: raise Exception(f'exception in {exp}')

class Environment:
    def __init__(self, parent=None):
        self.parent = parent
        self.vars = {}
        self.stk  = []

    def __repr__(self):
        s = repr(self.vars)
        if self.parent:
            return repr(self.parent) + '\n' + s
        return s

    def fork(self):
        return Environment(self)

    def __contains__(self, name):
        if name in self.vars:
            return True
        elif self.parent is None:
            return False
        else: return name in self.parent

    def __getitem__(self, name):
        if name in self.vars:
            return self.vars[name]
        elif self.parent is None:
            return NONE
        else: return self.parent[name]

    def __setitem__(self, name, value):
        if self.parent is not None and name in self.parent:
            self.parent[name] = value
        else: self.vars[name] = value

class Range():
    def __init__(self, low, upper, inclusive=(False, False), step=1):
        if upper < low: step *= -1
        self.bounds = low, upper
        self.inclusive = inclusive
        self.step = step

    def __repr__(self):
        lo, up = self.bounds
        match(self.inclusive):
            case (True,  True ): return f'[{lo},{up}]'
            case (False, True ): return f']{lo},{up}]'
            case (True,  False): return f'[{lo},{up}['
            case (False, False): return f']{lo},{up}['

    def __iter__(self):
        lo, up = self.bounds
        if not self.inclusive[0]: lo += self.step
        if self.inclusive[1]: up += self.step
        return iter(range(lo, up, self.step))

    def __contains__(self, x):
        (lo, up), inc = self.bounds, self.inclusive
        if self.step < 0:
            lo, up = up, lo
            inc = [inc[1], inc[0]]
        return lo <= x <= up and (lo < x or inc[0]) and (x < up or inc[1])

class Array(list):
    def __init__(self, arr, dic):
        super().__init__(arr)
        self.dic = dic

    def __repr__(self):
        s = super().__repr__()[1:-1]
        if len(self.dic) > 0:
            if len(s) > 0:
                s += ', '
            s += repr(self.dic)[1:-1]
        return '(' + s + ')'

    def __getitem__(self, name):
        if isinstance(name, str):
            if name == '':
                return Array(self.dic.values(), {})
            return self.dic[name]
        return super().__getitem__(name)

    def __setitem__(self, name, value):
        if isinstance(name, str):
            if name == '':
                raise Exception("key can't be empty string")
            self.dic[name] = value
        else: super().__setitem__(name, value)

class Lambda():
    def __init__(self, env, arg, fun, stk=False):
        self.env = env
        self.arg = arg
        self.fun = fun
        self.stk = stk

    def __repr__(self):
        head = '(' + ','.join(self.arg)  + ')' + ('$' if self.stk else '') + ' -> '
        if not verbose:
            return head + '{...}'
        return repr(self.env) + '\n' + head + repr(self.fun)

    def __iter__(self):
        return iter([self.env, self.arg, self.fun, self.stk])

    def len(self):
        return len(self.arg) if not self.stk else float('inf')

def language(s):
    global reserved
    has_run = False
    if not has_run:
        from deep_translator import GoogleTranslator
        has_run = True
    gt = GoogleTranslator(source='auto', target=s)
    for token in restoken:
        reserved[gt.translate(token).replace(' ', '_').lower()] = token

#   *----...-*
#   |  MAIN  |
#   *--------*

if __name__ == '__main__':
    import os, readline, argparse, sys, traceback, re

    argp = argparse.ArgumentParser(prog=sys.argv[0],
        description='ICC25 Interpreter\nInterpretation und Compilation von Computerprogrammen 2025\nHochschule Bonn-Rhein-Sieg',
        epilog='󱤹 Raphael Schönefeld', formatter_class=argparse.RawTextHelpFormatter)
    argp.add_argument('-i', '--input', help='ICC25 source code')
    argp.add_argument('-l', '--language')
    argp.add_argument('-sl', '--supported-languages', action='store_true')
    argp.add_argument('-t', '--write-tables', action='store_true')
    argp.add_argument('-v', '--verbose', action='store_true')
    args = argp.parse_args()
    verbose = args.verbose
    parser = yacc(start='exp', debug=args.verbose, write_tables=args.write_tables)

    if args.supported_languages:
        try:
            from deep_translator import GoogleTranslator
            for lang in GoogleTranslator().get_supported_languages(): print(lang)
        except:
            print("try: pip install deep-translator")
        sys.exit(0)
    
    env = Environment()
    if args.input:
        print(load(args.input))
        sys.exit(0)

    if args.language:
        language(args.language)
        
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

    while True:
        try:
            src = input('\x1b[0;32m' + '>>> ' + '\x1b[0m')
            while src.count('#') % 2 != 0 or src.count('{') != src.count('}'):# or src.count(':') > src.count('.') - len(re.findall(r'\b\d+\.\d+\b', src)):
                src += '\n' + input('\x1b[0;32m' + '... ' + '\x1b[0m')
            if src == '': continue
            if src.strip() == 'help':
                print('\n'.join([f'{value:5} = {key}' for key, value in reserved.items() if key not in engwords.keys()]))
                print(env)
                continue
            result = parser.parse(src)
            if verbose: print(result)
            if result: print('\x1b[0;34m' + repr(eval(result, env)) + '\x1b[0m')
        except EOFError: exit()
        except Exception as e: print('\x1b[0;31m' + (traceback.format_exc() if verbose else repr(e) + '\n') + '\x1b[0m', end='')
        except KeyboardInterrupt as e: print('\n\x1b[0;31m' + repr(e) + '\x1b[0m')
