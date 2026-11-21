from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama




def main():
    print('Hello from longchain')
    information = """
        William Bradley Pitt (born December 18, 1963) is an American actor and film producer. In a film career spanning more than thirty-five years, Pitt has received numerous accolades, including two Academy Awards, two British Academy Film Awards, two Golden Globe Awards, two Primetime Emmy Awards, and one Volpi Cup. His films as a leading actor have grossed over $7.5 billion worldwide.[2][3]

Pitt first gained recognition as a cowboy hitchhiker in the Ridley Scott road film Thelma & Louise (1991). Pitt emerged as a star taking on leading roles in films such as the drama A River Runs Through It (1992), the western Legends of the Fall (1994), the horror film Interview with the Vampire (1994), the crime thriller Seven (1995), the cult film Fight Club (1999), and the crime comedy Snatch (2000). He cemented his leading man status by starring in blockbusters, such as the Ocean's film trilogy (2001–2007), Troy (2004) and Mr. & Mrs. Smith (2005).[4] His other major successes include Inglourious Basterds (2009), World War Z (2013), Bullet Train (2022) and F1 (2025).

Pitt earned Academy Award nominations for his performances in the science fiction drama 12 Monkeys (1995), the fantasy romance The Curious Case of Benjamin Button (2008) and the sports drama Moneyball (2011). For his portrayal of a stuntman in Quentin Tarantino's Once Upon a Time in Hollywood (2019), he won the Academy Award for Best Supporting Actor, a role which he reprised in The Further Mis-Adventures of Cliff Booth (2026). He also starred in acclaimed films such as Babel (2006), The Assassination of Jesse James by the Coward Robert Ford (2007), Burn After Reading (2008), The Tree of Life (2011), Fury (2014), The Big Short (2015), Ad Astra (2019) and Babylon (2022).

In 2001, Pitt co-founded the production company Plan B Entertainment.[5] As a producer, he won the Academy Award for Best Picture for 12 Years a Slave (2013) and was nominated for Moneyball (2011) and The Big Short (2015). An influential figure in popular culture, Pitt appeared on Forbes' annual Celebrity 100 list from 2006 to 2008, and the Time 100 list in 2007. Regarded as a sex symbol, Pitt was named People's Sexiest Man Alive in 1995 and 2000. Pitt's relationships have been subject to widespread media attention, particularly his marriages to actresses Jennifer Aniston and Angelina Jolie.
    """

    summary_template = f"""
    given information {information} about a person I want you to create:
    1. A short summary
    2. 2 interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=['information'], template=summary_template,
    )

    #llm = ChatOllama(temperature=0, model='gpt-oss:20b')
    llm = ChatOllama(temperature=0, model='gemma3:270m')
    #llm = ChatOpenAI(temperature=0, model='gpt-5')
    chain = summary_prompt_template | llm
    response = chain.invoke(input={'information': information})
    print(response.content)
    print('ok')

if __name__ == '__main__':
    main()