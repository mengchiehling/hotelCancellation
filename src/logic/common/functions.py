import re
from typing import Tuple, Optional


def parenthesis_striped(ingredient: str) -> Tuple[str, Optional[str]]:
    '''
    strip parenthesis from the ingredient name

    Args:
        ingredient:
    Returns:

    '''

    name_1 = ingredient

    parenthesis_remove_regex = re.compile('[\w\s_]+(\([^)]+\))')
    '''
    \( : match an opening parentheses
    ( : begin capturing group
    [^)]+: match one or more non ) characters
    ) : end capturing group
    \) : match closing parentheses
    '''
    group = parenthesis_remove_regex.match(ingredient)
    if group:
        chemical_list = [e for e in ingredient.replace(group[1], "").split(" ") if e != '']
        name_1 = " ".join(chemical_list)

    return name_1