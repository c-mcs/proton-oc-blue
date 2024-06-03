import mesa
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
        self.oc_role = None
        self.oc_boss = None
        self.oc_subordinates = None
        self.family_role = None
        row = weighted_one_of(self.model.age_gender_dist, lambda x: x[-1],
                                    self.model.random)  # select a row from our age_gender distribution
        self.birth_tick = 0 - row[0] * self.model.ticks_per_year
        self.ticks = 0
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

    def get_all_relatives(self):
        # Combine all the specified neighbor sets into a single set to avoid repetition
        unique_neighbors = set(self.neighbors['household']) | set(self.neighbors['sibling']) | \
                        set(self.neighbors['offspring']) | set(self.neighbors['parent']) | \
                        set(self.neighbors['partner'])
        return unique_neighbors
    
    def get_neighbor_list(self, net_name: str):
        """
        Given the name of a network, this method returns a list of agents within the network.
        If the network is empty, it returns an empty list.
        :param net_name: str, the network name
        :return: list, return an empty list if the network is empty
        """
        agent_net = self.neighbors.get(net_name)
        if len(agent_net) > 0:
            return list(agent_net)
        else:
            return []
    
    def get_all_agents(self):
        """
        Given a list of network names, this method returns a set of all agents
        within these networks, without repetitions.
        :param network_names: list, the list of network names
        :return: set, a set of all agents in all networks
        """
        all_agents = set()
        for net_name in self.network_names:
            agents = self.get_neighbor_list(net_name)
            all_agents.update(agents)
        return all_agents
    
    def remove_from_model_graphs(self):
        for family in self.model.families:
            if self in family:
                family.remove(self)
                break
        if self in self.model.friends_graph:
            self.model.friends_graph.remove_node(self)
        if self in self.model.crime_communities_graph:
            self.model.crime_communities_graph.remove_node(self)

    def remove_from_household(self) -> None:
        """
        This method removes the agent from household, keeping the networks consistent.
        Modify the Person.neighbors attribute in-place
        :return: None
        """
        for member in self.neighbors.get("household").copy():
            if self in member.neighbors.get("household"):
                member.neighbors.get("household").remove(self)
                self.neighbors.get("household").remove(member)
                
    def p_mortality(self):
        if self.age in self.model.mortality_table:
            p = self.model.mortality_table[self.age][self.gender_is_male] / self.model.ticks_per_year
        elif self.age > max(self.model.mortality_table):
            p = 1
        return p
    
    def die(self):
        neighbors = self.get_all_agents()
        for agent in neighbors:
            for network in agent.network_names:
                if self in agent.neighbors.get(network):
                    agent.neighbors.get(network).remove(self)
        self.model.schedule.remove(self)
        self.remove_from_model_graphs()
