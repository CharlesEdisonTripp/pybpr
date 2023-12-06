from dataclasses import dataclass, replace
from typing import Any, Iterable, Iterator, Tuple, Union


@dataclass(slots=True, frozen=True, eq=True)
class Interaction:
    subject: Any # the actor
    verb: Any # the action
    object: Any # the target of the action
    timestamp:Any # the time of the interaction

    def subjectless(self) -> 'Interaction':
        return replace(self, subject=None)
    
    def verbless(self) -> 'Interaction':
        return replace(self, verb=None)
    
    def objectless(self) -> 'Interaction':
        return replace(self, object=None)
    
    def negative(self) -> 'Interaction':
        return replace(self, verb  = not self.verb)


Interactions = Union[Iterable[Interaction], Iterator[Interaction]]