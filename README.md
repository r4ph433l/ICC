# Values
| name | regex | example |
| --- | --- | --- |
| Decimal | `0\|[1-9][0-9]*` | `0`, `19`, `999` |
| Hexadecimal | `0x(0\|[1-9A-F][0-9A-F]*)` | `0x0`, `0x13`, `0x3E7` |
| Binary | `0b(0\|1[0-1]*)` | `0b0`, `0b10011`, `0b1111100111` |
| Float | `0\|[1-9][0-9]*\.[0-9]*[1-9]\|0` | `0.0`, `0.13`, `99.9` |
| Complex | `<any of the above>j` | `0j`, `0x13j`, `99.9j` |
| String | `".*"` | `"hi"` |
# Lists
lists are just comma separated expressions e.g. `1,2,3` -> `(1,2,3)`

empty list: `()`

list containing one item: `1,` -> `(1)`

if you want nested lists you can do `1,(2,3)` -> `(1,(2,3))` and let precedence do the rest

or `(1,2,3),,,` -> `(((1,2,3)))`
# Intervals
```
([])exp, exp([])
```
like mathematic intervals - brackets pointing outwards means that the value is excluded

can be used in a for loop

`exp in ([]) exp, exp ([])` returns `1` if the value (even floating point values) is inside the interval or `0` if not
# Operators
## Arithmetic
| operator | usage | python |
| --- | --- | --- |
| `+` | `a + b` |  |
| `+` | `+ a` | `abs(a)` |
| `-` | `a - b` |  |
| `-` | `- a` | `- a` |
| `*` | `a * b` |  |
| `\|` | `a \| b` | `a / b` |
| `/` | `a / b` | `ceil(a / b)` |
| `\` | `a \ b` | `floor(a / b)` |
| `**` | `a ** b` | |

| operator | alt | usage | python |
| --- | --- | --- | --- |
| `mod` | `≡` | `a mod b` | `a % b` |
| `and` | `∧` | `a and b` | `int(bool(a) and bool(b))` |
| `or`  | `∨` | `a or b` | `int(bool(a) or bool(b))` |
| `xor` | `⊻` | `a xor b` | `int(bool(a) == bool(b))` |
| `not` | `¬` | `not a` | `int(not bool(a))` |
## Comparisons
`<`, `>`, `<=`, `>=`, `==` and `!=` work the same as in python

mathematical comparison chains like `0 < a < 1` are also possible
## Assignments
| operator | usage | python |
| --- | --- | --- |
| `=` | `a = 1` | |
| `++` | `a++` | `a += 1` |
| `--` | `a--` | `a -= 1` |

any binary operator assignment combination like `mod=` are also possible
# Sequences
```
{exp; exp(;)}
```

also returns last expression as value
# Control Structures
```
if exp: exp (else exp).
? exp: exp (! exp).
```
```
for exp in IT: exp.
∀ exp ∈ IT: exp.
```
`IT` can be a list or an interval

you can also write `for 2i in [0,10]: echo i.` to print every even number between 0 and 10
```
while exp: exp.
⟲ exp: exp.
```
all structures return the last expression as value
# Lambda Functions
```
f = a,b -> a+b
f(2,3)
```
if you oversupply a function the arguments get added to a stack

to access the stack use `$exp`, `$0` returns the length of the stack

lambda functions also introduce lexical scoping
# Inbuild Functions
| function | alternative | usage | description |
| --- | --- | --- |
| `echo` | `♫` | `echo "hi"` | prints to stdout |
| `load` | `⊃` | `load "test.icc"` | executes code in given file |
| `eval` | `⊢` | `eval "print('hi')"` | executes code in python |

little trick: if you are on linux you can just `load "/dev/stdin"` to input data
# Syntax translation
dont use it

...

you can add a "Langbang" to the top of your file so your syntax gets translated: `#?italian`
