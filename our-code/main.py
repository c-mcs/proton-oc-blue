import time
from ocmodel import CrimeModel
import matplotlib.pyplot as plt
import networkx as nx
import random
import os
from collections import defaultdict

local = True

def print_report(model, current_directory, tag, save_to_file=False, age_category_width=15):
    report_lines = []

    # Function to print details of a family
    def print_family(family):
        family_lines = ["Family:"]
        for member in family:
            sex = "Male" if member.gender_is_male else "Female"
            role = member.family_role.capitalize()
            family_lines.append(f"  Member ID: {member.unique_id}, Age: {member.age}, Sex: {sex}, Role: {role}")
        family_lines.append("\n" + "-"*50 + "\n")
        return family_lines

    # Function to print details of an agent and their friends
    def print_agent_and_friends(agent):
        agent_lines = []
        sex = "Male" if agent.gender_is_male else "Female"
        agent_lines.append(f"Agent ID: {agent.unique_id}, Age: {agent.age}, Sex: {sex}, Education: {agent.education_level}, Work Status: {agent.job_level}")
        agent_lines.append("Friends:")
        for friend in agent.neighbors['friendship']:
            friend_sex = "Male" if friend.gender_is_male else "Female"
            agent_lines.append(f"  Friend ID: {friend.unique_id}, Age: {friend.age}, Sex: {friend_sex}, Education: {friend.education_level}, Work Status: {friend.job_level}")
        agent_lines.append("\n" + "-"*50 + "\n")
        return agent_lines

    # Sample 10% of the families
    num_families_sample = max(1, int(len(model.families) * 0.1))
    sampled_families = random.sample(model.families, num_families_sample)

    # Collect family details
    for family in sampled_families:
        report_lines.extend(print_family(family))

    # Sample 10% of the agents
    agents = list(model.schedule.agents)
    num_agents_sample = max(1, int(len(agents) * 0.1))
    print(f"NUMAGENTS {len(agents)}")
    sampled_agents = random.sample(agents, num_agents_sample)

    # Collect agent and friends details
    for agent in sampled_agents:
        report_lines.extend(print_agent_and_friends(agent))

    # Dictionary to store age categories and gender data
    age_gender_categories = defaultdict(lambda: {'Male': {'total_friends': 0, 'total_avg_age': 0, 'count': 0},
                                                 'Female': {'total_friends': 0, 'total_avg_age': 0, 'count': 0}})

    # Collect agent data for the entire dataset
    friend_distribution = defaultdict(lambda: {'Male': 0, 'Female': 0})
    for agent in agents:
        # Calculate friends statistics by age category and gender
        if agent.age // age_category_width * age_category_width < 75:
            age_category = (agent.age // age_category_width) * age_category_width
        else:
            age_category = 75
        
        gender = 'Male' if agent.gender_is_male else 'Female'
        friends = agent.neighbors['friendship']

        if friends:
            avg_age_of_friends = sum(friend.age for friend in friends) / len(friends)
            age_gender_categories[age_category][gender]['total_avg_age'] += avg_age_of_friends
            age_gender_categories[age_category][gender]['count'] += 1
            age_gender_categories[age_category][gender]['total_friends'] += len(friends)
        
        # Calculate friend distribution
        num_friends = len(friends)
        friend_distribution[num_friends][gender] += 1

    # Print average number of friends and average of average age of friends by age category and gender
    report_lines.append("Average number of friends and average of average age of friends by age category and gender:")
    report_lines.append("Age Category | Gender | Avg. Friends | Avg. of Avg. Age of Friends")
    report_lines.append("-" * 70)
    
    for age_category in sorted(age_gender_categories.keys()):
        for gender in ['Male', 'Female']:
            total_friends = age_gender_categories[age_category][gender]['total_friends']
            total_avg_age = age_gender_categories[age_category][gender]['total_avg_age']
            count = age_gender_categories[age_category][gender]['count']
            if count > 0:
                avg_friends = total_friends / count
                avg_of_avg_age_of_friends = total_avg_age / count
                if age_category == 75:
                    report_lines.append(f"    {int(age_category):3d}+     | {gender:6} | {avg_friends:12.2f} | {avg_of_avg_age_of_friends:22.2f}")
                else:
                    report_lines.append(f"    {int(age_category):3d} -{int(age_category + age_category_width - 1):3d} | {gender:6} | {avg_friends:12.2f} | {avg_of_avg_age_of_friends:22.2f}")
            else:
                if age_category == 75:
                    report_lines.append(f"    {int(age_category):3d}+     | {gender:6} | {'No data':>12} | {'No data':>22}")
                else:
                    report_lines.append(f"    {int(age_category):3d} -{int(age_category + age_category_width - 1):3d} | {gender:6} | {'No data':>12} | {'No data':>22}")
        report_lines.append("\n")

    # Print percentual distribution of each number of friends disaggregated by gender
    report_lines.append("Percentual distribution of number of friends by gender:")
    report_lines.append("Number of Friends | Male (%) | Female (%)")
    report_lines.append("-" * 50)
    
    total_male = sum(friend_distribution[num]['Male'] for num in friend_distribution)
    total_female = sum(friend_distribution[num]['Female'] for num in friend_distribution)

    for num_friends in sorted(friend_distribution.keys()):
        male_percentage = (friend_distribution[num_friends]['Male'] / total_male * 100) if total_male > 0 else 0
        female_percentage = (friend_distribution[num_friends]['Female'] / total_female * 100) if total_female > 0 else 0
        report_lines.append(f"       {num_friends:4d}       |  {male_percentage:6.2f}  |  {female_percentage:7.2f}")

    # Output report
    report_str = "\n".join(report_lines)

    crime_report = "\n"
    for head in model.mafia_heads:
        crime_report += f"Head: {head.unique_id}. Age: {head.age}, Sex: {head.gender_is_male}, Crime P: {head.propensity}\n"
        for el in [f"Member: {m.unique_id}. Age: {m.age}, Sex: {m.gender_is_male}, Crime P: {m.propensity}\n" for m in head.oc_subordinates]:
            crime_report += el
        crime_report += "\n"
    report_str += crime_report
    if save_to_file:
        filename = f"report{tag}.txt"
        report_path = os.path.join(current_directory, filename)
        with open(report_path, "w") as file:
            file.write(report_str)
    else:
        print(report_str)



def show_friends_graph(model):
    plt.subplot(1, 1, 1)  # Only one subplot
    nx.draw(model.friends_graph, with_labels=False, node_size=20, edge_color='w', node_color='b', alpha=0.7)
    plt.title('Friends Graph')
    plt.gcf().set_facecolor('black')
    plt.show()

def show_family_graph(model):
    family_graph = nx.Graph()

    for family in model.families:
        for member in family:
            family_graph.add_node(member.unique_id, label=member.name)
            for neighbor in member.neighbors['household']:
                family_graph.add_edge(member.unique_id, neighbor.unique_id)

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(family_graph)
    nx.draw(family_graph, pos, with_labels=True, node_size=200, font_size=8)
    plt.title('Family Networks')
    plt.show()

def show_graphs(model):
    start_time = time.time()

    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))

    # Friendship graph
    pos_friends = nx.spring_layout(model.friends_graph)
    nx.draw(model.friends_graph, pos_friends, ax=axes[0], with_labels=False, node_size=20, edge_color='black', node_color='b', alpha=0.7)
    axes[0].set_title('Friendship Network')
    axes[0].set_facecolor('black')

    # Family graph

    family_graph = nx.Graph()
    for family in model.families:
        for member in family:
            family_graph.add_node(member.unique_id)
            for relative in member.get_all_relatives():
                family_graph.add_edge(member.unique_id, relative.unique_id)

    
    pos_family = nx.spring_layout(family_graph)
    nx.draw(family_graph, pos_family, ax=axes[1], node_size=20, font_size=8)
    axes[1].set_title('Family Network')
    axes[1].set_facecolor('black')

    # Crime communities graph
    pos_crime_communities = nx.circular_layout(model.crime_communities_graph)
    nx.draw(model.crime_communities_graph, pos_crime_communities, ax=axes[2], with_labels=False, node_size=20, edge_color='red', node_color='r', alpha=0.7)
    axes[2].set_title('Crime Communities')
    axes[2].set_facecolor('black')

    print("--- %s seconds --- (To show graph)" % (time.time() - start_time))
    plt.show()

if __name__ == "__main__":
    if local:
        current_directory = os.path.dirname(os.path.abspath(__file__))
    start_time = time.time()
    m = CrimeModel(5000, current_directory = current_directory)
    print("--- %s seconds --- (To calculate)" % (time.time() - start_time))
    print_report(m, current_directory, "1", True)
    j = 0
    for i in range(360):
        m.step()
        if i % 72 == 0:
            j+=1
            print_report(m, current_directory, str(j), True)
    show_graphs(m)
    