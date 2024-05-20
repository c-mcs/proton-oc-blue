import mesa
import names
from extras import *

class Person(mesa.Agent):
    network_names = [
        'sibling',
        'offspring',
        'parent',
        'partner',
        'household',
        'friendship',
        'criminal']

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.neighbors = self.networks_init()

    def init_person(self) -> None:
        """
        This method modifies the attributes of the person instance based on the model's
        stats_tables as part of the initial setup of the model agents.
        :return: None
        """
        self.oc_member = False
        self.family_role = None
        self.name = names.get_full_name()
        row = weighted_one_of(self.model.age_gender_dist, lambda x: x[-1],
                                    self.model.random)  # select a row from our age_gender distribution
        self.birth_tick = 0 - row[0] * self.model.ticks_per_year  # ...and set age... =
        #self.age = (self.model.tick - self.birth_tick) / self.model.ticks_per_year # to fix?
        self.age = row[0]
        self.gender_is_male = bool(row[1])  # ...and gender according to values in that row.
        self.retired = self.age >= self.model.retirement_age  # persons older than retirement_age are retired
        # education level is chosen, job and wealth follow in a conditioned sequence
        
        # Set education level based on age
        if self.age < 13:
            self.education_level = 1.0
        elif self.age < 18:
            self.education_level = min(2.0, pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random))
        elif self.age < 21:
            self.education_level = min(3.0, pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random))
        else:
            self.education_level = pick_from_pair_list(self.model.edu[self.gender_is_male], self.model.random)
            
        self.propensity = self.model.lognormal(self.model.nat_propensity_m, self.model.nat_propensity_sigma)
        self.job_level = 0
        self.wealth_level = 0 # job e wealth level le mettiamo con un'altra funzione ma è bene ricordarci che sono variabili dell'agente
        self.num_co_offenses = {}

    def step(self):
        return None

    def networks_init(self):
        return {i: set() for i in Person.network_names}

    def make_friendship_link(self, asker):
        self.neighbors.get("friendship").add(asker)
        asker.neighbors.get("friendship").add(self)

    def make_partner_link(self, asker) -> None:
        """
        Create a two-way partner link in-place
        :param asker: Person
        :return: None
        """
        self.neighbors.get("partner").add(asker)
        asker.neighbors.get("partner").add(self)

    def make_parent_offsprings_link(self, asker):
        """
        Create a link between parent and offspring. Askers are the offspring.
        :param asker: Union[List[Person], Person]
        :return: None
        """
        if type(asker) == list:
            for person in asker:
                self.neighbors.get("offspring").add(person)
                person.neighbors.get("parent").add(self)
        else:
            self.neighbors.get("offspring").add(asker)
            asker.neighbors.get("parent").add(self)

    def add_sibling_link(self, targets):
        """
        Create a two-way sibling links in-place
        :param targets: List[Person]
        :return: None
        """
        for x in targets:
            if x != self:
                self.neighbors.get("sibling").add(x)
                x.neighbors.get("sibling").add(self)

    def make_household_link(self, targets):
        """
        Create a two-way household link in-place
        :param targets: List[Person]
        :return: None
        """
        for x in targets:
            if x != self:
                self.neighbors.get("household").add(x)
                x.neighbors.get("household").add(self)
    
    
    def add_criminal_link(self, asker):
        """
        Create a two-way criminal links in-place and update the connection weight
        :param asker: Person
        :return: None
        """
        self.neighbors.get("criminal").add(asker)
        self.num_co_offenses[asker] = 1
        asker.neighbors.get("criminal").add(self)
        asker.num_co_offenses[self] = 1
