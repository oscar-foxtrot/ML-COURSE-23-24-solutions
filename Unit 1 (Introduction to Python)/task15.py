from typing import List


def hello(name: str='') -> str:
    return "Hello!" if name == '' else f"Hello, {name}!"


def int_to_roman(num: int) -> str:  # no foolproofness
    retval = ''  # gonna return this
    romans = ['I', 'V', 'X', 'L', 'C', 'D', 'M']  # all this for uniformity
    curiteration = len(str(num)) * 2 - 2  # for iterating over the "romans"
    for i in str(num):
        if int(i) in range(0, 4):
            retval += romans[curiteration] * int(i)
        elif int(i) == 4:
            retval += romans[curiteration] + romans[curiteration + 1]
        elif int(i) in range(5, 9):
            retval += romans[curiteration + 1] + romans[curiteration] * (int(i) - 5)
        else:  # here i == '9'
            retval += romans[curiteration] + romans[curiteration + 2]
        curiteration -= 2
    return retval


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return '' 
    res = ''
    for i in range(len(strs_input)):
        strs_input[i] = strs_input[i].lstrip()
    for i in range(0, min(len(strs_input[0]), len(strs_input[i]))):
        for elem in strs_input[1:]:
            if elem[i] != strs_input[0][i]:
                return res 
        res += strs_input[0][i]
    return res
                         

def primes() -> int:
    #  Here could be a time-efficient algorithm with some memory
    #  usage as the trade-off, but then the generator loses its purpose
    def isprime(j_arg: int) -> bool:
        for i in range(2, j_arg):
            if j_arg % i == 0:
                return False
        return True
    i = 2
    while True:
        if isprime(i):
            yield i
        i += 1
    

class BankCard:
    def __init__(self, total_sum: int=0, balance_limit: int=None):  # I'd make total_sum private but im not touching the prototypes
        self.total_sum = total_sum
        self.balance_limit = balance_limit
        
    def __call__(self, sum_spent: int) -> None:
        if self.total_sum < sum_spent:
            raise ValueError(f"Not enough money to spend {sum_spent} dollars.")
        else:
            self.total_sum -= sum_spent
            print(f"You spent {sum_spent} dollars.")
            
    def __str__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, \
            None if self.balance_limit is None or other.balance_limit is None \
                else max([self.balance_limit, other.balance_limit]))
    
    @property
    def balance(self) -> int:
        if self.balance_limit == 0:
            raise ValueError('Balance check limits exceeded.')
        else:
            if self.balance_limit is not None:
                self.balance_limit -= 1
            return self.total_sum
        
    def put(self, sum_put: int) -> None:
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")
    