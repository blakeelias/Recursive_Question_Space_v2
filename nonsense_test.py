from openai import OpenAI
import os
import logging
from dotenv import load_dotenv

def generate_completion(client, prompt: str, system_role: str) -> str:
    """Generate a completion with error handling"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Make sure to use your desired model
            messages=[
                {"role": "system", "content": system_role},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"Error in API call: {e}")

def load_prompt(prompt_dir: str) -> str:
    """Load the nonsense check prompt template"""
    try:
        file_path = os.path.join(prompt_dir, "nonsense_prompt.txt")
        with open(file_path, 'r') as file:
            return file.read().strip()
    except Exception as e:
        raise Exception(f"Error loading nonsense prompt: {e}")

def check_nonsense(client, prompt_template: str, content: str, system_role: str = "You are a philosophical critic evaluating statements for meaningfulness.") -> float:
    """
    Check if content is meaningful or nonsense
    Returns: float score from 0-100, where higher scores indicate more nonsensical content
    """
    prompt = prompt_template.format(content=content)

    try:
        result = generate_completion(client, prompt, system_role)
        try:
            score = float(result.strip())
            return score
        except ValueError:
            logging.error(f"Could not parse nonsense score from response: {result}")
            return 0.0
    except Exception as e:
        raise Exception(f"Error generating nonsense check: {e}")

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Your OpenAI API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    prompt_dir = "./prompts"  # Make sure this points to your prompts directory
    
    # Initialize only what we need
    client = OpenAI(api_key=api_key)
    prompt_template = load_prompt(prompt_dir)
    
    # Test cases
    test_cases = [
        "This view posits that knowledge requires three conditions: a belief must be true, it must be justified, and it must be held by the individual. Justification involves having good reasons or evidence for the belief, thereby ensuring that it is not merely a lucky coincidence that the belief is true. This perspective emphasizes the importance of the processes by which beliefs are formed and the contexts in which they are held, arguing that these elements are integral to the nature of knowledge, distinguishing it from mere true belief which may lack sufficient grounding.",
        "This view maintains that knowledge involves not only a true belief and the ability to produce true beliefs but also the requirement of justification, where justification includes rational support or evidence for the belief. Knowledge is understood as a mental state where the individual possesses a true belief that is backed by sufficient reasoning, reflecting an awareness of the justification process as a central aspect of epistemology. By incorporating justification, this perspective emphasizes that it is not enough just to have the ability to produce true beliefs; one must also be able to justify why those beliefs are true in order for them to count as knowledge.",
        "Reliabilism posits that a belief counts as knowledge if it is true and formed through a reliable cognitive process, one that consistently produces true beliefs across a range of circumstances. This view emphasizes the importance of the processes that lead to belief formation and argues that as long as the process is generally reliable, an individual can possess knowledge, even if there are instances of luck resulting in true beliefs. The focus on the reliability of the cognitive mechanisms implies that knowledge is not simply about isolated instances of justified true belief but rather about the overall effectiveness of the methods used to arrive at those beliefs, thereby allowing for a more nuanced understanding of knowledge that accommodates different situations without falling into the traps set by Gettier problems.",
        "Contextualism posits that the truth conditions for knowledge claims can vary based on the context in which they are assessed, allowing for a flexible understanding of knowledge that accounts for different perspectives. This view holds that what counts as knowledge is contingent upon the standards applicable in particular situations, suggesting that the context informs the evaluation of a belief's truth and justification. By recognizing that various contexts may call for different criterions of knowledge, contextualism aims to provide a more universally applicable account of knowledge that mitigates the ambiguity tied to individual intellectual virtues, enabling more consistent judgments across differing scenarios.",
        "Critical rationalism asserts that knowledge is not a matter of justification based on consensus or agreement, but rather a process of conjecture and refutation. In this view, the pursuit of knowledge involves proposing hypotheses that can be critically tested and possibly falsified, allowing for a form of dialogue that focuses on the scrutiny of ideas instead of requiring shared values or frameworks. This perspective encourages engagement across different epistemic communities through the rigorous testing and challenging of claims, promoting a dynamic discourse where conflicting ideas can inform and improve understanding. The commitment to critique and openness to challenging one's own views provides a mechanism for knowledge evaluation that remains productive even in the face of diverse and potentially opposing beliefs.",
        "Constructive empiricism asserts that the goal of scientific theories is to accurately account for observable phenomena, rather than to provide a true depiction of an underlying reality. Central commitments involve the belief that a theory is considered successful if it can suitably describe and predict observable events, regardless of whether it corresponds to an unobservable underlying structure. This view emphasizes the agreement among scientists as valuable when forming a shared theoretical framework, which can consolidate consensus around empirical adequacy. Constructive empiricism thus allows for a critical yet consensual engagement with knowledge claims, where theories are evaluated based on their utility and coherence with observed data, rather than purely on their refutation potential.",
        "Fallibilism posits that while we may possess knowledge, this knowledge is inherently tentative and open to change. The view emphasizes that no belief or theory can ever be definitively proven true; instead, they can only be supported through evidence that may change with further scrutiny. Central commitments include the idea that critical engagement with ideas is essential for growth and improvement of knowledge. Fallibilism encourages the idea that revising our understanding is a natural part of the scientific process, allowing for a dynamic interplay of ideas where flaws can be identified and corrected without the necessity of consensus.",
        "This view posits that knowledge requires not only true belief but also justification that provides a sound basis for holding that belief. The justified belief must be supported by adequate evidence or reasons, which differentiate it from mere true belief obtained by chance. The justification element emphasizes a deeper cognitive engagement with the belief, suggesting that a person must have a rational basis for their belief that goes beyond mere reliability of the process used to arrive at it. A belief is considered knowledge not just when it is true and produced by a reliable method, but also when the believer has sufficient reasons or evidence that secure the belief's epistemic status, ensuring it is not merely a lucky guess.",
        "This perspective maintains that for a belief to qualify as knowledge, it must meet three criteria: it must be true, the person must believe it, and it must be justified. The justification aspect underscores the necessity for beliefs to be backed by reasons or evidence, thereby addressing cases where true beliefs arise without adequate justification. This view creates a comprehensive framework in which the reliability of belief-forming processes plays a role, but it insists that knowledge is ultimately contingent on meeting the additional requirement of justification. In this sense, a belief cannot simply be a product of reliable methods; it must also be accompanied by a rationale that substantiates the belief in a non-coincidental manner.",
        "This view asserts that knowledge requires not only a belief that is true but also justification for that belief. Justification refers to a rational account or evidence that supports the truth of the belief, thus linking internal coherence to external reality. In this framework, knowledge is constituted by beliefs that are true and for which individuals have adequate justification, forming a robust connection between belief and the actual state of the world. This perspective emphasizes the necessity of aligning coherent beliefs with factual accuracy through justification processes, thus ensuring that knowledge claims are anchored both in internal consistency and their correspondence to reality.",
        "Critical Rationalism argues that knowledge advances through a process of conjectures and refutations, suggesting that knowledge claims must be subjected to rigorous criticism and testing rather than accepted based on consensus or coherence alone. This view emphasizes the role of scientific inquiry and the necessity of actively seeking out and addressing conflicting theories as a means to sharpen and improve understanding. By valuing critical engagement and open debate, Critical Rationalism provides a robust framework for resolving conflicting perspectives, as it fosters an environment where competing viewpoints must justify themselves through evidence and reasoned argument, thus contributing to the progression of knowledge.",
        "This view holds that there are objective truths about the world that can be discovered through observation, reasoning, and empirical evidence. While acknowledging that individual perspectives may color interpretations of knowledge, it maintains that there exist facts about the world that remain true regardless of personal or cultural beliefs. Realist Epistemology emphasizes a correspondence theory of truth, whereby statements are considered true if they accurately depict the external reality. This framework permits discourse and debate about knowledge claims while asserting that objective standards of verification exist. Hence, this view underlines the importance of an external, accessible reality that can be reliably known, despite the subjective lenses individuals may hold.",
        "Constructivism holds that knowledge is not simply found in self-evident beliefs but is actively constructed by individuals as they engage with their social environments. This view emphasizes that understanding arises from the interplay of personal experiences and communal dialogue, with the recognition that beliefs can evolve through discussion and collaboration. Knowledge is therefore seen as a dynamic and contextual process, influenced by cultural background, prior knowledge, and social frameworks. Rather than relying solely on self-evidence, constructivism recognizes the importance of relationships and societal contexts in shaping what we come to know.",
        "This view asserts that for a belief to qualify as knowledge, it must not only be justified but also true. It emphasizes that knowledge is fundamentally linked to facts about the world, and any belief that is false cannot be considered knowledge, despite its usefulness in particular contexts. The central commitment of this view is that genuine knowledge requires an alignment with reality; thus, the truth of a belief serves as a non-negotiable criterion. In this framework, practical outcomes or contextual effectiveness cannot substitute for the requirement that knowledge must be grounded in truth, ensuring that false beliefs, even if they yield practical success, are excluded from being classified as knowledge, maintaining the integrity of knowledge as a reliable concept."
        
    
    
    
    ]
    
    # Run tests
    
    scores = []

    for _ in range(1):
        for i, test in enumerate(test_cases, 1):
            try:
                score = check_nonsense(client, prompt_template, test)
                scores.append(score)
                print(f"\nTest case {i}:")
                print(f"Nonsense Score: {score}")
            except Exception as e:
                print(f"Error processing test case {i}: {e}")

    if scores:
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"\nStatistics over 100 iterations:")
        print(f"Average Nonsense Score: {avg_score}")
        print(f"Minimum Nonsense Score: {min_score}")
        print(f"Maximum Nonsense Score: {max_score}")
    else:
        print("No scores to report.")

if __name__ == "__main__":
    main()