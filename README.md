# ICC Language Syntax Guide
## Comments
- Block comments are enclosed with `#...#`
```icc
# This is a comment #
```
## Literals
### Numbers
- Supports decimal, binary (`0b`), hexadecimal (`0x`), and complex (`j`) numbers:
```icc
42
0b1010
0xFF
3.14
2j
```
### Strings
- Enclosed in double quotes:
```icc
"Hello world!"
```
## Operators
### Arithmetic

| Operator     | Meaning        |
| ------------ | -------------- |
| `+`          | Addition       |
| `-`          | Subtraction    |
| `*`          | Multiplication |
| `\|`         | Division       |
| `/`          | Ceil Division  |
| `\`          | Floor Division |
| `**`         | Exponentiation |
| `mod` or `≡` | Modulo         |
### Boolean Logic

| Operator     | Meaning |
| ------------ | ------- |
| `or`  or `∨` | OR      |
| `and` or `∧` | AND     |
| `xor` or `⊻` | XOR     |
| `not` or `¬` | NOT     |
## Comparison

|Operator|Meaning|
|---|---|
|`==`|Equal|
|`!=`|Not equal|
|`<, >`|Less/Greater|
|`<=, >=`|Less/Greater-equal|
Comparisons can also be grouped like this: `0 < b < 1`
## Variables
```icc
x = 10
x++        # Increment
x--        # Decrement 
a[0] = 5   # Array element assignment
```
## Arrays and Ranges
```icc
1, 2, 3                 # Array
()                      # Empty Array
(1,)                    # Array containing 1 element
(1, 2, (a = 4))         # Array with named elements
a[0]                    # access element
a["a"]                  # access named element
a[""]                   # access Array of named elements
[1, 10]                 # Inclusive range
]1, 10[                 # Exclusive range
```
## Control Flow
### If/Else
```icc
if cond : exp else exp.

? cond : exp ! exp.
```
### For Loops
```icc
for i in 1,2,3,4 : exp.
for 2i in [1,10] : exp.        # with stepsize

∀ i ∈ 1,2,3,4 : exp.
∀ 2i ∈ [1,10] : exp.
```
### While Loops
```icc
while exp : exp.

⟲ exp : exp.
```
## Functions & Lambdas
```icc
(a,b)  -> a+b                               # Lambda
()$ -> for i in ]0,$0]: echo $i             # Lambda with stack support
func(1, 2)                                  # Function call
```
# Buildin Functions

| Function       | Meaning        |
| -------------- | -------------- |
| `echo` or `♫`  | print          |
| `read` or `𝄽` | input          |
| `load` or `⊃`  | import         |
| `eval` or `⊢`  | eval in python |
| `size` or `‖`  | length         |
```icc
♫ "Hello, world!"
⊃ "script.icc"
⊢ "2 + 2"
```
## Stack Access
```icc
$1        # Top stack value
$0        # Stack size
```
## Language Translation
You can add `#?italian` to the top of your file and every keyword will be translated to this language. This needs Internet, because its powered by GoogleTranslate :D

## Termination

Multiple statements can be grouped with `;`, and blocks are enclosed in `{}`.

```icc
{
  a = 2;
  b = 3;
  a + b
}.
```

## Example

```icc
# Calculate factorial #
fact = (n) -> {
  ? n == 0 : 1 ! n * fact(n - 1)
}.
```
