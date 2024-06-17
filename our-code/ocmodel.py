import os
import numpy as np
import mesa
import networkx as nx
from Person import Person
from extras import *
import random
from itertools import combinations, chain

class CrimeModel(mesa.Model):
    def __init__(self, N, current_directory, model_params = {"no-params":"empty"}, agent_params = {"no-params":"empty"}):
        super().__init__()
        self.current_directory = current_directory
        self.number_weddings = 0
        self.number_deceased = 0
        self.families = list()
        self.mafia_heads = set()
        self.tick = 0
        self.ticks_per_year = 12
        self.nat_propensity_m: float = 1.0  # 0.1 -> 10
        self.nat_propensity_sigma: float = 0.25  # 0.1 -> 10.0
        self.nat_propensity_threshold: float = 1.0  # 0.1 -> 2.0
        self.education_modifier = 1.0
        self.parameters = model_params
        self.agent_params = agent_params
        self.initial_agents = N
        self.schedule = mesa.time.RandomActivation(self)
        self.running = True
        self.random = np.random.default_rng(1)
        self.random_state = random.randint(1,100)
        self.retirement_age = 64
        self.friends_graph = nx.Graph()
        self.crime_communities_graph = nx.Graph()
        self.meta_graph = nx.Graph()
        self.number_arrests_per_year: int = 30  # 0 -> 100
        self.number_crimes_yearly_per10k: int = 2000 
        self.arrest_rate = self.number_arrests_per_year / self.ticks_per_year / self.number_crimes_yearly_per10k / 10000 * self.initial_agents
        self.num_oc_persons: int = 30  # 2 -> 200
        self.num_oc_families: int = 8  # 1 -> 50
        self.this_is_a_big_crime: int = 3
        self.good_guy_threshold: float = 0.6
        self.big_crime_from_small_fish: int = 0  # checking anomalous crimes
        self.number_offspring_recruited_this_tick: int = 0
        self.number_law_interventions_this_tick = 0
        self.people_jailed = 0
        self.number_crimes = 0
        self.number_born = 0
        self.punishment_length: int = 1  # 0.5 -> 2
        self.max_accomplice_radius: int = 2  # 2 -> 4
        self.oc_embeddedness_radius: int = 2  # 1 -> 4
        self.crime_size_fails = 0

        self.datacollector = mesa.datacollection.DataCollector(
            model_reporters={
                "Steps": lambda m: m.schedule.steps,
            }
        )
        self.setup()

    def setup(self):
        self.init_data_employed()
        self.init_agents()
        self.assign_jobs_and_wealth()
        self.generate_households()
        self.cleanup_unfit_individuals()
        self.setup_siblings()
        self.generate_friends_network()
        self.calculate_crime_multiplier()
        self.calculate_criminal_tendency()
        self.setup_oc_groups2()
        for agent in self.schedule.agents:
            if not agent.gender_is_male and agent.get_neighbor_list("offspring"):
                agent.number_of_children = len(agent.get_neighbor_list("offspring"))
    
        self.calculate_criminal_tendency()
        self.calculate_crime_multiplier() 

    def step(self):
        self.tick += 1
        self.number_law_interventions_this_tick = 0 # TODO

        for agent in self.schedule.agents:
            agent.ticks += 1
            agent.num_crimes_committed_this_tick = 0 # TODO
            if agent.ticks % 12 == 0:
                agent.age += 1

        if self.tick % 12 == 0:
            self.calculate_criminal_tendency()
            self.calculate_crime_multiplier() 
            self.update_education_wealth_job()
            self.update_friendships()
            self.add_immigrants()
        
        self.reset_oc_embeddedness()
        self.commit_crimes()
        self.make_baby()
        self.retire_persons()
        self.wedding()
        self.make_people_die()


    def init_data_employed(self):
        self.age_gender_dist = read_csv_data("initial_age_gender_dist", self.current_directory).values.tolist()
        self.head_age_dist = df_to_dict(read_csv_data("head_age_dist_by_household_size", self.current_directory))
        self.proportion_of_male_singles_by_age = df_to_dict(read_csv_data("proportion_of_male_singles_by_age", self.current_directory))
        self.hh_type_dist = df_to_dict(read_csv_data("household_type_dist_by_age", self.current_directory))
        self.partner_age_dist = df_to_dict(read_csv_data("partner_age_dist", self.current_directory))
        self.children_age_dist = df_to_dict(read_csv_data("children_age_dist", self.current_directory))
        self.p_single_father = read_csv_data("proportion_single_fathers", self.current_directory)
        self.edu = df_to_dict(read_csv_data("edu", self.current_directory))
        self.mortality_table = df_to_dict(read_csv_data("initial_mortality_rates", self.current_directory), extra_depth=True)
        self.work_status_by_edu_lvl = df_to_dict(read_csv_data("work_status_by_edu_lvl", self.current_directory))
        self.wealth_quintile_by_work_status = df_to_dict(read_csv_data("wealth_quintile_by_work_status", self.current_directory))
        self.c_range_by_age_and_sex = df_to_lists(read_csv_data("crime_rate_by_gender_and_age_range", self.current_directory))
        marriage = read_csv_data("marriages_stats", self.current_directory)
        self.number_weddings_mean = marriage['mean_marriages'][0]
        self.number_weddings_sd = marriage['std_marriages'][0]
        self.fertility_table = df_to_dict(read_csv_data("initial_fertility_rates", self.current_directory), extra_depth=True)
        self.punishment_length_data = read_csv_data("conviction_length", self.current_directory)
        self.male_punishment_length = df_to_lists(
            self.punishment_length_data[["months", "M"]], split_row=False)
        self.female_punishment_length = df_to_lists(
            self.punishment_length_data[["months", "F"]], split_row=False)
        self.num_co_offenders_dist = df_to_lists(
            read_csv_data("num_co_offenders_dist", self.current_directory), split_row=False)
    def cleanup_unfit_individuals(self):
        for a in self.schedule.agents:
            if a.family_role == None:
                self.schedule.agents.remove(a)

    def household_sizes(self, size: int):
        """
        Loads a table with a probability distribution of household size and calculates household
        based on initial agents
        :param size: int, population size
        :return: list, the sizes of household
        """
        hh_size_dist = read_csv_data("household_size_dist", self.current_directory).values
        sizes = []
        current_sum = 0
        while current_sum < size:
            hh_size = pick_from_pair_list(hh_size_dist, self.random)
            if current_sum + hh_size <= size:
                sizes.append(hh_size)
                current_sum += hh_size
        sizes.sort(reverse=True)
        return sizes

    def lognormal(self, mu, sigma) -> float:
        return np.exp(mu + sigma * self.random.normal())
    
    def init_agents(self, immigrants = None):
        if immigrants:
            N = immigrants
        else:
            N = self.initial_agents
        for i in range(N):
            new_agent = Person(i, self)
            new_agent.init_person()
            self.schedule.add(new_agent)
            if immigrants:
                new_agent.immigrant = True

    def calculate_similarity(self, agent, potential_friend):
        age_diff = abs(agent.age - potential_friend.age)
        gender_diff = int(agent.gender_is_male != potential_friend.gender_is_male)
        education_diff = abs(agent.education_level - potential_friend.education_level)
        wealth_diff = abs(agent.wealth_level - potential_friend.wealth_level)

        age_weight = 3
        gender_weight = 2.0
        education_weight = 1.0
        wealth_weight = 1.0

        # Check if either the agent or the potential friend is a minor and if the age difference exceeds 4 years
        if (agent.age < 18 or potential_friend.age < 18) and age_diff > 4:
            similarity_score = 100.0  # Set a high similarity score to indicate low compatibility
        else:
            similarity_score = (age_weight * age_diff + 
                                gender_weight * gender_diff + 
                                education_weight * education_diff + 
                                wealth_weight * wealth_diff)
        return similarity_score

    def generate_friends_network(self):
        agents = list(self.schedule.agents)
        self.random.shuffle(agents) 
        
        # Calculate mean friend count based on age of agents
        typical_number_of_friends = 3
        mean_friend_counts = [typical_number_of_friends - (agent.age / 99) * 1.5 for agent in agents]  # Adjust this scaling factor as needed
        # Generate friend counts for each agent using a normal distribution with scaled mean
        friend_counts = [self.random.normal(loc=mean, scale=2) for mean in mean_friend_counts]
        friend_counts = np.clip(friend_counts, 0, typical_number_of_friends*2).astype(int)  # Ensure friend counts are between 0 and 6
        sample_size = min(int(self.initial_agents / 3), len(agents) - 1)  # Limit the sample size to 1/3(N) or total agents - 1
        for agent, num_friends in zip(agents, friend_counts):
            excluded_neighbors = agent.get_all_relatives()

            potential_friends = [a for a in agents if a != agent and a not in excluded_neighbors]
            if len(potential_friends) > sample_size:
                potential_friends = self.random.choice(potential_friends, sample_size, replace=False)
            
            if len(potential_friends) == 0:
                continue
            # Filter potential friends based on age difference exceeding 4 years
            potential_friends = [friend for friend in potential_friends if self.calculate_similarity(agent, friend) < 100.0]
            if len(potential_friends) == 0:
                continue
            
            similarity_scores = np.array([self.calculate_similarity(agent, a) for a in potential_friends])
            inv_similarity_scores = 1 / (similarity_scores + 1e-6)
            probabilities = inv_similarity_scores / inv_similarity_scores.sum()
            actual_num_friends = min(num_friends, len(potential_friends))  # Ensure actual_num_friends does not exceed potential friends
            
            if actual_num_friends != 0:
                r = random.random()
                if agent.gender_is_male and r < 0.20:
                    continue
                if not agent.gender_is_male and r < 0.05:
                    continue

            # Adjust number of friends based on gender bias with probability
            if agent.gender_is_male and actual_num_friends != 0 and self.random.random() < 0.8:
                actual_num_friends -= 1
                
            selected_friends = self.random.choice(potential_friends, size=actual_num_friends, replace=False, p=probabilities)
            for friend in selected_friends:
                agent.make_friendship_link(friend)
                
        # Create the friends graph
        self.friends_graph = nx.Graph()
        for agent in self.schedule.agents:
            self.friends_graph.add_node(agent.unique_id)
            for friend in agent.neighbors['friendship']:
                self.friends_graph.add_edge(agent.unique_id, friend.unique_id)



    def generate_households(self) -> None:
        """
        This procedure aggregates eligible agents into households based on the tables
        (ProtonOC.self.head_age_dist, ProtonOC.proportion_of_male_singles_by_age,
        ProtonOC.hh_type_dist, ProtonOC.partner_age_dist, ProtonOC.children_age_dist,
        ProtonOC.p_single_father) and mostly follows the third algorithm from Gargiulo et al. 2010
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0008828)

        :return: None
        """
        population = self.schedule.agents
        self.hh_size = self.household_sizes(self.initial_agents)
        complex_hh_sizes = list()
        max_attempts_by_size = 50
        attempts_list = list()

        for size in self.hh_size:
            success, hh_members, nb_attempts = self.attempt_to_form_household(size, population, max_attempts_by_size)
            if success:
                self.finalize_successful_household(hh_members)
                attempts_list.append(nb_attempts)
            else:
                complex_hh_sizes.append(size)

        self.remove_unsuccessful_agents(population)
        return

    def attempt_to_form_household(self, size, population, max_attempts_by_size):
        success = False
        nb_attempts = 0

        while not success and nb_attempts < max_attempts_by_size:
            nb_attempts += 1
            hh_members = self.generate_household_members(size, population)

            if len(hh_members) == size:
                success = True

        return success, hh_members, nb_attempts

    def generate_household_members(self, size, population):
        hh_members = []

        head_age = pick_from_pair_list(self.head_age_dist[size], self.random)
        if size == 1:
            male_wanted = self.random.random() < self.proportion_of_male_singles_by_age[head_age]
            head = pick_from_population_pool_by_age_and_gender(head_age, male_wanted, population, self.random)
            if head:
                head.family_role = "head"
                hh_members.append(head)
        else:
            hh_members = self.generate_complex_household_members(size, head_age, population)

        return hh_members

    def generate_complex_household_members(self, size, head_age, population):
        hh_members = []

        hh_type = pick_from_pair_list(self.hh_type_dist[head_age], self.random)
        male_head = self.random.random() < float(self.p_single_father.columns.to_list()[0]) if hh_type == "single_parent" else True
        mother_age = pick_from_pair_list(self.partner_age_dist[head_age], self.random) if male_head else head_age

        head = pick_from_population_pool_by_age_and_gender(head_age, male_head, population, self.random)
        if head:
            hh_members.append(head)
            head.family_role = "head"

            if hh_type == "couple":
                mother = pick_from_population_pool_by_age_and_gender(mother_age, False, population, self.random)
                if mother:
                    hh_members.append(mother)
                    mother.family_role = "partner"

            hh_members.extend(self.generate_children(int(size - len(hh_members)), mother_age, population))

        return hh_members

    def generate_children(self, num_children, mother_age, population):
        children = []
        if num_children in self.children_age_dist and mother_age in self.children_age_dist[num_children]:
            children_age_dist_for_mother = self.children_age_dist[num_children][mother_age]
            for _ in range(num_children):
                child_age = pick_from_pair_list(children_age_dist_for_mother, self.random)
                child = pick_from_population_pool_by_age(child_age, population, self.random)
                if child:
                    child.family_role = "child"
                    children.append(child)

        return children

    def finalize_successful_household(self, hh_members):
        family_wealth_level = hh_members[0].wealth_level

        if len(hh_members) > 1 and hh_members[1].family_role == "partner":
            hh_members[0].make_partner_link(hh_members[1])
            couple = hh_members[:2]
            offsprings = hh_members[2:]

            for partner in couple:
                partner.make_parent_offsprings_link(offsprings)
            for sibling in offsprings:
                sibling.add_sibling_link(offsprings)

        for member in hh_members:
            member.make_household_link(hh_members)
            member.wealth_level = family_wealth_level

        self.families.append(hh_members)

    def remove_unsuccessful_agents(self, population):
        print(f"Removing {len(population)} agents.")
        for m in population:
            self.schedule.agents.remove(m)


    def setup_siblings(self) -> None:
        """
        Right now, during setup, links between agents are only those within households, between
        friends and related to the school. At this stage of the standard setup, agents are linked
        through "siblings" links outside the household. To simulate agents who have left the
        original household, agents who have children are taken and "sibling" links are created
        taking care not to create incestuous relationships.

        :return: None
        """
        agent_left_household = [p for p in self.schedule.agents if
                                p.neighbors.get('offspring')]
        # simulates people who left the original household.
        for agent in agent_left_household:
            num_siblings = self.random.poisson(0.5)
            # 0.5 -> the number of links is N^3 agents, so let's keep this low at this stage links
            # with other persons are only relatives inside households and friends.
            candidates = [c for c in agent_left_household
                          if c not in agent.neighbors.get("household")
                          and abs(agent.age - c.age) < 5 and c != agent]
            # remove couples from candidates and their neighborhoods (siblings)
            if len(candidates) >= 50:
                candidates = self.random.choice(candidates, 50, replace=False).tolist()
            while len(candidates) > 0 and list_contains_problems(agent, candidates):
                # trouble should exist, or check-all-siblings would fail
                potential_trouble = [x for x in candidates if agent.get_neighbor_list("partner")]
                trouble = self.random.choice(potential_trouble)
                candidates.remove(trouble)
            targets = [agent] + self.random.choice(candidates,
                                                   min(len(candidates), num_siblings)).tolist()
            for sib in targets:
                if sib in agent_left_household:
                    agent_left_household.remove(sib)
            for target in targets:
                target.add_sibling_link(targets)
                # this is a good place to remind that the number of links in the sibling link
                # neighbors is not the "number of brothers and sisters"
                # because, for example, 4 brothers = 6 links.
            other_targets = targets + [s for c in targets for s in c.neighbors.get('sibling')]
            for target in other_targets:
                target.add_sibling_link(other_targets)

    def assign_jobs_and_wealth(self, single_agent = None) -> None:
        """
        This procedure modifies the job_level and wealth_level attributes of agents in-place.
        This is just a first assignment, and will be modified first by the multiplier then
        by adding neet status.

        :return: None
        """
        
        permuted_set = self.random.permuted(self.schedule.agents)
        if single_agent:
            permuted_set = [single_agent]
        for agent in permuted_set:
            if agent.age > 16:
                agent.job_level = pick_from_pair_list(
                    self.work_status_by_edu_lvl[agent.education_level][agent.gender_is_male],
                    self.random)
                agent.wealth_level = pick_from_pair_list(
                    self.wealth_quintile_by_work_status[agent.job_level][agent.gender_is_male],
                    self.random)
            else:
                agent.job_level = 1.0
                agent.wealth_level = 1.0

    def calculate_crime_multiplier(self) -> None:
            """
            Based on ProtonOC.c_range_by_age_and_sex this procedure modifies in-place
            the attribute ProtonOc.crime_multiplier
            :return: None
            """
            total_crime = 0
            for line in self.c_range_by_age_and_sex:
                people_in_cell = [agent for agent in self.schedule.agents if line[0][1] <
                                agent.age <= line[1][0] and agent.gender_is_male == line[0][0]]
                n_of_crimes = line[1][1] * len(people_in_cell)
                total_crime += n_of_crimes
            self.crime_multiplier = \
                self.number_crimes_yearly_per10k / 10000 * len(self.schedule.agents) / total_crime

    def calculate_criminal_tendency(self) -> None:
        """
        Based on the ProtonOC.c_range_by_age_and_sex distribution, this function calculates and
        assigns to each agent a value representing the criminal tendency. It modifies the attribute
        Person.criminal_tendency in-place.

        :return: None
        """
        for line in self.c_range_by_age_and_sex:
            # the line variable is composed as follows:
            # [[bool(gender_is_male), int(minimum age range)],
            # [int(maximum age range), float(c value)]]
            subpop = [agent for agent in self.schedule.agents if
                      line[0][1] <= agent.age <= line[1][0] and agent.gender_is_male == line[0][0]]
            if subpop:
                c = line[1][1]
                # c is the cell value. Now we calculate criminal-tendency with the factors.
                for agent in subpop:
                    agent.criminal_tendency = c
                    agent.update_criminal_tendency()
                # then derive the correction epsilon by solving
                # $\sum_{i} ( c f_i + \epsilon ) = \sum_i c$
                epsilon = c - np.mean([agent.criminal_tendency for agent in subpop])
                for agent in subpop:
                    agent.criminal_tendency += epsilon
        #if self.intervention_is_on() and self.facilitator_repression:
            #self.calc_correction_for_non_facilitators() # Possibilità di inserire qui una policy
    
    def setup_oc_groups2(self) -> None:
        """
        This procedure creates "criminal" type links within the families, based on the criminal
        tendency of the agents. In case the agents within the families are not enough, new members
        are taken from outside, ensuring that members of the same household belong to the same mafia family.
        :return: None
        """
        # OC members are scaled down if we don't have 10K agents
        scaled_num_oc_families = int(np.ceil(
            self.num_oc_families * self.initial_agents / 10000 * self.num_oc_persons / 30))
        scaled_num_oc_persons = int(np.ceil(
            self.num_oc_persons * self.initial_agents / 10000))
        members_per_family = max(1, int(scaled_num_oc_persons / scaled_num_oc_families))  # Ensure at least 1 member per family
        oc_family_heads = weighted_n_of(scaled_num_oc_families, [agent for agent in self.schedule.agents if agent.gender_is_male],
                                        lambda x: x.criminal_tendency, self.random)
        graphs = []
        for head in oc_family_heads:
            head.oc_member = True
            head.oc_role = "boss"
            self.mafia_heads.add(head)
            
            # Collect family members ensuring household constraint
            candidates = self.collect_candidates(head)
            
            # Fill up the family with internal candidates first
            cosca = self.assign_family_members(head, candidates, members_per_family)
            
            # If not enough members, get additional members from outside
            if len(cosca) < members_per_family:
                missing_n = members_per_family - len(cosca)
                broader_family_candidates = self.collect_out_of_family_candidates(head, cosca, missing_n)
                additional_members = weighted_n_of(missing_n, broader_family_candidates, lambda x: x.criminal_tendency, self.random)
                cosca.update(self.assign_additional_members(additional_members, cosca, head, fill = True))
            if len(cosca) < members_per_family:
                missing_n = members_per_family - len(cosca)
                out_of_family_candidates = [agent for agent in self.schedule.agents
                                        if not agent.oc_member and agent.age >= 18]
                out_of_family_candidates = weighted_n_of(missing_n, out_of_family_candidates, lambda x: x.criminal_tendency, self.random)
                cosca.update(self.assign_additional_members(out_of_family_candidates, cosca, head, fill = True))
            
            cosca_graph = nx.Graph()
            for cosca_member in cosca:
                cosca_member.oc_member = True
                cosca_member.oc_role = "soldier"
                cosca_member.oc_boss = head
                cosca_graph.add_node(cosca_member)
                cosca_graph.add_edge(head, cosca_member)
            
            head.oc_subordinates = cosca
            cosca_graph.add_node(head)
            graphs.append(cosca_graph)

        self.crime_communities_graph = nx.compose_all(graphs)
        for node1, node2 in combinations(self.mafia_heads, 2):
            self.crime_communities_graph.add_edge(node1, node2)

    def collect_candidates(self, head):
        candidates = set()
        for relative in head.get_all_relatives():
            if relative.age >= 18:
                candidates.add(relative)
            candidates.update(m for m in relative.neighbors["household"] if m.age >= 18 and not m.oc_member)
        return candidates

    def assign_family_members(self, head, candidates, members_per_family):
        cosca = set()
        members_in_families = weighted_n_of(min(members_per_family, len(candidates)), candidates, lambda x: x.criminal_tendency, self.random)
        for member in members_in_families:
            if member != head:  # Ensure head is not included
                cosca.add(member)
        return cosca

    def collect_out_of_family_candidates(self, head, cosca, missing_n):
        out_of_family_candidates = set()
        for member in cosca:
            for agent in member.get_all_relatives():
                if any(a.oc_boss != head for a in agent.neighbors["household"]):
                    continue
                if agent.age >= 18 and agent.gender_is_male:
                    out_of_family_candidates.add(agent)
                    if len(out_of_family_candidates) >= missing_n:
                        return out_of_family_candidates
        return out_of_family_candidates

    def assign_additional_members(self, additional_members, cosca, head, fill=False):
        for candidate in additional_members:
            if candidate != head:  # Ensure head is not included
                cosca.add(candidate)
                if fill:
                    for household_member in candidate.neighbors["household"]:
                        if household_member.age >= 18:
                            cosca.add(household_member)
        return cosca

    def make_people_die(self) -> None:
        dead_agents = list()
        permuted = self.random.permuted(self.schedule.agents)
        for agent in permuted:
            if self.random.random() < agent.p_mortality() or agent.age > 119:
                dead_agents.append(agent)
                self.number_deceased += 1
        for agent in dead_agents:
            if agent.oc_role == "boss":
                if len(agent.oc_subordinates) > 0:
                    oldest = max(agent.oc_subordinates, key=lambda person: person.age)
                    oldest.oc_role = "boss"
                    oldest.oc_subordinates = agent.oc_subordinates
                    oldest.oc_subordinates.remove(oldest)
                    for sub in oldest.oc_subordinates:
                        sub.oc_boss = oldest
            agent.die()
            del agent

    def wedding(self) -> None:
        corrected_weddings_mean = (self.number_weddings_mean * len(self.schedule.agents) / 1000) / 12
        num_wedding_this_month = self.random.poisson(corrected_weddings_mean)
        marriable = [agent for agent in self.schedule.agents if 25 < agent.age < 55
                    and not agent.neighbors.get("partner")]
        
        while num_wedding_this_month > 0 and len(marriable) > 1:
            ego = self.random.choice(marriable)
            friends = ego.neighbors.get("friendship")
            
            if not friends:
                continue
            
            partner = None
            best_similarity_score = float('inf')
            
            for friend in friends:
                if 25 < friend.age < 55 and not friend.neighbors.get("partner"):
                    similarity_score = self.calculate_similarity_wedding(ego, friend)
                    if similarity_score < best_similarity_score:
                        best_similarity_score = similarity_score
                        partner = friend
            if partner:
                for agent in [ego, partner]:
                    agent.remove_from_household()
                ego.neighbors.get("household").add(partner)
                partner.neighbors.get("household").add(ego)
                ego.neighbors.get("partner").add(partner)
                partner.neighbors.get("partner").add(ego)
                marriable.remove(partner)
                num_wedding_this_month -= 1
                self.number_weddings += 1
            marriable.remove(ego)

    def calculate_similarity_wedding(self, agent, potential_partner):
            age_diff = abs(agent.age - potential_partner.age)
            gender_diff = int(agent.gender_is_male != potential_partner.gender_is_male)
            education_diff = abs(agent.education_level - potential_partner.education_level)
            wealth_diff = abs(agent.wealth_level - potential_partner.wealth_level)

            age_weight = 3.0
            gender_weight = 2.0
            education_weight = 1.0
            wealth_weight = 1.0

            # Favor older male for female agents and younger female for male agents
            if agent.gender_is_male:
                if potential_partner.age >= agent.age:
                    age_preference_score = max(0, 8 - age_diff)  # Favor younger female partners
                else:
                    age_preference_score = max(0, 8 - age_diff) / 2  # Reduce score for older female partners
            else:
                if potential_partner.age <= agent.age:
                    age_preference_score = max(0, 8 - age_diff)  # Favor older male partners
                else:
                    age_preference_score = max(0, 8 - age_diff) / 2  # Reduce score for younger male partners

            # Calculate the similarity score
            similarity_score = (
                age_weight * (8 - age_preference_score) + 
                gender_weight * gender_diff + 
                education_weight * education_diff + 
                wealth_weight * wealth_diff
            )

            return similarity_score


    def make_baby(self):
        for agent in [agent for agent in self.schedule.agents if
                          14 <= agent.age <= 50 and not agent.gender_is_male]:
                if self.random.random() < agent.p_fertility():
                    agent.init_baby()


    def retire_persons(self) -> None:
        """
        Agents that reach the self.retirement_age are retired.
        :return: None
        """
        to_retire = [agent for agent in self.schedule.agents
                     if agent.age >= self.retirement_age and not agent.retired]
        for agent in to_retire:
            agent.retired = True

    def update_education_wealth_job(self):
        agents_filtered = [agent for agent in self.schedule.agents if agent.age == 13 or agent.age >= 18]
        for agent in agents_filtered:
            just_changed = False
            if agent.age == 13 or agent.age == 18 or agent.age == 21:
                education = agent.education_level
                agent.education_level = min(agent.education_level, pick_from_pair_list(self.edu[agent.gender_is_male], self.random))
                if education != agent.education_level:
                    just_changed = True
            if agent.age > 16 and just_changed:
                self.assign_jobs_and_wealth(agent)

    def update_friendships(self):
        X = 0.1  # 10% probability
        update_share = 0.1  # 10% of the friendship pool

        agents = list(self.schedule.agents)
        self.random.shuffle(agents) 

        for agent in agents:
            if self.random.random() < X:
                # Determine the number of friendships to be renovated
                current_friends = list(agent.neighbors['friendship'])
                current_friends_n = len(current_friends)
                if not current_friends:
                    current_friends_n = random.randint(1,4)  # Skip the agent if it has no friends

                num_friends_to_renovate = max(1, int(current_friends_n * update_share))
                
                # Break existing friendships
                if num_friends_to_renovate > 0 and current_friends:
                    friends_to_remove = self.random.choice(current_friends, num_friends_to_renovate, replace=False)
                    for friend in friends_to_remove:
                        agent.break_friendship_link(friend)

                # Create new friendships using the similarity process
                excluded_neighbors = agent.get_all_relatives()
                potential_friends = [a for a in agents if a != agent and a not in excluded_neighbors]
                sample_size = min(int(self.initial_agents / 3), len(potential_friends))  # Limit the sample size to 1/3(N) or total agents - 1

                if len(potential_friends) > sample_size:
                    potential_friends = self.random.choice(potential_friends, sample_size, replace=False)

                if len(potential_friends) == 0:
                    continue

                potential_friends = [friend for friend in potential_friends if self.calculate_similarity(agent, friend) < 100.0]
                if len(potential_friends) == 0:
                    continue
                
                similarity_scores = np.array([self.calculate_similarity(agent, a) for a in potential_friends])
                inv_similarity_scores = 1 / (similarity_scores + 1e-6)
                probabilities = inv_similarity_scores / inv_similarity_scores.sum()
                actual_num_friends = min(num_friends_to_renovate, len(potential_friends))  # Ensure actual_num_friends does not exceed potential friends

                if actual_num_friends > 0:
                    selected_friends = self.random.choice(potential_friends, size=actual_num_friends, replace=False, p=probabilities)
                    for friend in selected_friends:
                        agent.make_friendship_link(friend)
                    
        # Update the friends graph
        self.friends_graph.clear()
        for agent in self.schedule.agents:
            self.friends_graph.add_node(agent.unique_id)
            for friend in agent.neighbors['friendship']:
                self.friends_graph.add_edge(agent.unique_id, friend.unique_id)

    def add_immigrants(self):
        n_immigrants = int(self.initial_agents * 0.01)
        self.init_agents(n_immigrants)
            
    def commit_crimes(self) -> None:
        """
        This procedure is central in the model, allowing agents to find accomplices and commit
        crimes. Based on the table ProtonOC.c_range_by_age_and_sex, the number of crimes and the
        subset of the agents who commit them is selected. For each crime a single agent is selected
        and if necessary activates the procedure that allows the agent to find accomplices.
        Criminal groups are append within the list co_offender_groups.

        :return: None
        """
        co_offender_groups = list()
        co_offender_started_by_oc = list()
        for cell, value in self.c_range_by_age_and_sex:
            people_in_cell = [agent for agent in self.schedule.agents if
                              cell[1] <= agent.age <= value[0]
                              and agent.gender_is_male == cell[0]]
            target_n_of_crimes = \
                value[1] * len(people_in_cell) / self.ticks_per_year * self.crime_multiplier
            for _target in np.arange(np.round(target_n_of_crimes)):
                self.number_crimes += 1
                starter = weighted_one_of(people_in_cell,
                                              lambda x: x.criminal_tendency,
                                              self.random)
                number_of_accomplices = self.number_of_accomplices()
                accomplices = starter.find_accomplices(number_of_accomplices)
                # this takes care of facilitators as well.
                co_offender_groups.append(accomplices)
                if starter.oc_member:
                    co_offender_started_by_oc.append((accomplices, starter))
                # check for big crimes started from a normal guy
                if len(accomplices) > self.this_is_a_big_crime \
                        and starter.criminal_tendency < self.good_guy_threshold:
                    self.big_crime_from_small_fish += 1
        for co_offender_group in co_offender_groups:
            commit_crime(co_offender_group) # sta in extras
        for co_offenders_by_OC, starter in co_offender_started_by_oc:
            for agent in [agent for agent in co_offenders_by_OC if not agent.oc_member]:
                agent.new_recruit = self.tick
                agent.oc_member = True
                if starter.oc_role == "boss":
                    agent.oc_boss = starter
                    starter.oc_subordinates.add(agent)
                else:
                    agent.oc_boss = starter.oc_boss
                    starter.oc_boss.oc_subordinates.add(agent)
                agent.oc_role = "soldier"
                if agent.father:
                    if agent.father.oc_member:
                        self.number_offspring_recruited_this_tick += 1
        criminals = list(chain.from_iterable(co_offender_groups))
        if criminals:
            # no intervention active
            for criminal in criminals:
                criminal.arrest_weight = 1
            arrest_mod = self.number_arrests_per_year / self.ticks_per_year / 10000 * len(self.schedule.agents)
            target_n_of_arrest = np.floor(
                arrest_mod + 1
                if self.random.random() < (arrest_mod - np.floor(arrest_mod))
                else 0)
            for agent in weighted_n_of(target_n_of_arrest, criminals,
                                             lambda x: x.arrest_weight, self.random):
                agent.get_caught() # TODO

    def number_of_accomplices(self) -> int:
        """
        Pick a group size from ProtonOC.num_co_offenders_dist distribution and substract one to get
        the number of accomplices
        :return: int
        """
        return pick_from_pair_list(self.num_co_offenders_dist, self.random) - 1
    
    def update_meta_links(self, agents) -> None:
        """
        This method creates a new temporary graph that is used to colculate the
        oc_embeddedness of an agent.

        :param agents: Set[Person], the agentset
        :return: None
        """
        self.meta_graph = nx.Graph(seed=self.random_state)
        for agent in agents:
            self.meta_graph.add_node(agent.unique_id)
            for in_radius_agent in agent.agents_in_radius(
                    1):  # limit the context to the agents in the radius of interest
                self.meta_graph.add_node(in_radius_agent.unique_id)
                w = 0
                for net in Person.network_names:
                    if in_radius_agent in agent.neighbors.get(net):
                        if net == "criminal":
                            if in_radius_agent in agent.num_co_offenses:
                                w += agent.num_co_offenses[in_radius_agent]
                        else:
                            w += 1
                self.meta_graph.add_edge(agent.unique_id, in_radius_agent.unique_id, weight=1 / w)

    def reset_oc_embeddedness(self) -> None:
        """
        Reset the Person.cached_oc_embeddedness of all agents, this procedure is activated every
        tick before committing crimes.
        :return: None
        """
        for agent in self.schedule.agents:
            agent.cached_oc_embeddedness = None