from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Tuple, Union


@dataclass(slots=True, frozen=True, eq=True)
class Interaction:
    subject: Any # the actor
    verb: Any # the action
    object: Any # the target of the action

    def subjectless(self) -> 'Interaction':
        return Interaction(None, self.verb, self.object)
    
    def verbless(self) -> 'Interaction':
        return Interaction(self.subject, None, self.object)
    
    def objectless(self) -> 'Interaction':
        return Interaction(self.subject, self.verb, None)

    def make_positive_interaction(self) -> 'Interaction':
        return Interaction(None, self.verb, self.object)

    def make_negative_interaction(self) -> 'Interaction':
        return Interaction(None, self.verb, self.object)

    def make_binary_interactions(self) -> Tuple['Interaction', 'Interaction']:
        return self.make_positive_interaction(), self.make_negative_interaction()
    
    def negative(self) -> 'Interaction':
        return Interaction(self.subject, not self.verb, self.object)


Interactions = Union[Iterable[Interaction], Iterator[Interaction]]