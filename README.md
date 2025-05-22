# Values
| name | regex | example |
| --- | --- | --- |
| Decimal | `0\|[1-9][0-9]*` | `0`, `19`, `999` |
| Hexadecimal | `0x(0\|[1-9A-F][0-9A-F]*)` | `0x0`, `0x13`, `0x3E7` |
| Binary | `0b(0\|1[0-1]*)` | `0b0`, `0b10011`, `0b1111100111` |
| Float | `0\|[1-9][0-9]*\.[0-9]*[1-9]\|0` | `0.0`, `0.13`, `99.9` |
| Complex | `<any of the above>j` | `0j`, `0x13j`, `99.9j` |
# Operator
## Arithmetic
| operator | usage | python |
| --- | --- | --- |
| `+` | `a + b` |  |
| `-` | `a - b` |  |
| `*` | `a * b` |  |
| `\|` | `a | b` | `a / b` |
| `/` | `a / b` | `ceil(a / b)` |
| `\` | `a \ b` | `floor(a / b)` |
| `**` | `a ** b` | |
| `mod` | `a mod b` | `a % b` |
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
