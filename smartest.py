"""
Smartest generator + evaluator

Pasii algortimului (cumva)
1. un user cere intrebare pe un topic
2. modelul genereaza intrebare + solutie
    ex: pt N-queens (n fiind 4) modelul genereaza solutia {q1:2,q2:4,q3:1,q4:3}
        extrage raspunsul partial {q1:2;q3:1}
        genereaza intrebarea dupa un template asemanator cu "avand aceste date {input n, instances[]}, rezolva problema n-queens"
3. Userul raspunde (in ce mod vrea, gen poata sa spuna si "salut chat")
4. Modelul verifica daca raspunsul este sau nu valida 
5. Evalueaza raspunsul dat de user cu solutia calculata precedent
6. Da un scor acestui raspuns
"""

import json
import random
import re
from typing import Dict, Tuple, Optional

import sys
sys.path.append('.')



class QuestionGenerator:
    """Generează întrebări și păstrează soluțiile pentru evaluare"""
    
    def __init__(self):
        self.current_question = None
        self.current_solution = None
        self.current_metadata = None
    
    def solve_n_queens(self, n:int):
        """Cauta o rezolvare pentru N queens folosind Backtracking """


        #Functie care verifica daca pozitia linie coloana nu este "atacata" de vreo regina
        #astfel incat sa putem plasa o dama acolo
        def is_safe(board, linie, col):
            
            #verificarea pe coloana
            for i in range(linie):
                if board[i] == col:
                    return False
                
            #verificarea pe diagonala
            for i in range(linie):
                if abs(board[i] - col) == abs(i - linie): #|dif coloane| = |dif linii| 
                    return False
            return True
        

        #functia backtracking pe care o folosim
        # ca sa ajungem la UNA dintre solutii
        def backtrack(board, row):
            
            #cazu de baza
            if row == n:
                return True
            #randomizeaza modul in care v a vizita coloanele
            cols = list(range(n))
            random.shuffle(cols)


            for col in cols:
                if is_safe(board, row, col):#veriifca coloana by coloana daca este ok sa putem acolo regina
                    board[row] = col
                    #trece la urmatoarea linie recursiv
                    #daca functia returneaza true, propagam true mai mai sus
                    if backtrack(board, row + 1):  
                        return True
                    #elimina dama daca nu este ok
                    board[row] = -1
                    
            return False
        


        #za juiceee
        board = [-1] * n #lista marimea n plina de -1 
        if backtrack(board, 0):#aplicam backtracking si daca am returnat TRUE am gasit O solutie         
            return {f"Q{i+1}": board[i]+1 for i in range(n)}
        return None
    
    def generate_n_queens(self, difficulty='easy'):
        """Genereaza problema n-queens pe baza dificultatii din input (default=easy)"""
        # Mapam dificultatea care momentan este bazata doar pe marimea tablei si a nr de queen uri
        # TO DO cauta problema care nu se bazeaza doar pe marimea problemei dar si daca este posibil sa faca un pas inapoi
        size_map = {
            'easy': random.randint(4, 6),
            'medium': random.randint(7, 9),
            'hard': random.randint(10, 12)
        }
        size = size_map.get(difficulty, 8)
        
        # Genereaza solutia
        full_solution = self.solve_n_queens(size)
        #literalmente full_solution o sa fie  ori {} ori  sau {"qi": val ... } unde i < n si val este 
        
        if not full_solution:
            return None
        
        # faza unde creem solutia partiala 
        if difficulty=='easy':
            num_to_keep=max(1, size // 3)  
        else:
            num_to_keep=max(1, size // 4)

        queens_to_keep = random.sample(list(full_solution.keys()), num_to_keep)
        partial = {q: full_solution[q] for q in queens_to_keep}
        
        # Genereaza intrebarea dupa un template
        question = f"Complete the {size}-Queens problem. Current positions: {partial}. Where should the remaining queens be placed?"
        
        # Salvam intern intrebarea
        self.current_question = question
        self.current_solution = full_solution
        self.current_metadata = {
            'type': 'n-queens',
            'size': size,
            'partial': partial,
            'difficulty': difficulty
        }
        
        return question

    def generate_question(self, topic: str, difficulty: str = 'easy') -> str:
        """
        Genereaza intrebarea bazata pe un topic si o dificultatea
        
        Args:
            topic: 'n-queens', 'minmax', 'graph-coloring', etc.
            difficulty: 'easy', 'medium', 'hard'
        
        Returns:
            String cu întrebarea (soluția e păstrată intern)
        """
        topic = topic.lower().replace(' ', '-')
        
        if 'queen' in topic:
            return self.generate_n_queens(difficulty)
        else:
            return f"Scuze nu am fost trainuit pe {topic}"
    
    def get_solution(self) -> Dict:
        """returneaza solutia curenta si la ce pas este (for internal use)"""
        return {
            'solution': self.current_solution,
            'metadata': self.current_metadata
        }
class AnswerEvaluator:
    """Evalueaza raspunsurile userilor fata de solutia gasita"""
    
    def __init__(self, generator: QuestionGenerator):
        self.generator = generator
    
    def parse_n_queens_answer(self, answer: str) -> Optional[Dict]:
        """Extrage soluția N-Queens din răspunsul user-ului"""
        # nuj mom incearca sa caute un dictionar
        # dict_match = re.search(r'\{[^}]+\}', answer)
        # if dict_match:
        #     try:
        #         import ast
        #         return ast.literal_eval(dict_match.group(0))
        #     except:
        #         pass
        
        # incearca sa gaseasca perechi de tipul Q1:val Q2:val
        pattern = r'Q(\d+)[:\s=]+(\d+)'
        matches = re.findall(pattern, answer)
        if matches:
            return {f"Q{q}": int(v) for q, v in matches}
        
        return None
    
    def evaluate_n_queens(self, user_answer: str) -> Tuple[float, str]:
        """
        Evalueaza raspunsul primit de la user si returneaza scor + feedback
        ex: return  74.0/100, ai marcat 2/4 dame
            return 0.0/100 nu ai raspuns la intrebare
        """
        correct_solution = self.generator.current_solution
        metadata = self.generator.current_metadata
        
        if not correct_solution:
            return 0.0, "Nu este vreo intrebare activa"
        
        # trecem prin raspuns (parsing)
        #user_solution acum este un dictionar cu solutia data de user
        user_solution = self.parse_n_queens_answer(user_answer)
        

        #cazul cel mai simplu
        if not user_solution:
            return 0.0, "Could not parse your answer. Please provide queen positions like: {'Q1': 5, 'Q2': 3, ...}"
        
        #comparam dimenziunea 
        #cazum in care ca raspuns primim mai putine dame 
        size = metadata['size']
        if len(user_solution) != size:
            return 20.0, f"Raspuns incomplet! plasat: {len(user_solution)} queens dar aveai nevoie de: {size}."
        
        

        #Calculam scorul bazat raspuns
        #Adica calculam cate dame sunt puse corect, iar cele puse incorect le salvam in errors (benefic pt feedback)
        correct_count = 0
        errors = []
        
        for queen, pos in user_solution.items():
            if queen in correct_solution:
                if correct_solution[queen] == pos:
                    correct_count += 1
                else:
                    errors.append(f"{queen} should be at {correct_solution[queen]}, not {pos}")
        
        scor = (correct_count / size) * 100
        
        # "generam" feedbackul pe baza scorului
        if scor == 100:
            feedback = "Raspuns corect! Toate damele sunt puse corect"
        elif scor >= 80:
            feedback = f"Raspuns BUN \n{correct_count}/{size}. Cateva erori:\n"
            feedback += "\n".join(errors[:3])
        elif scor >= 50:
            feedback = f"Raspuns OK {correct_count}/{size}.Erorile:\n"
            feedback += "\n".join(errors[:5])
        else:
            feedback = f"Raspuns gresit!\n"
            feedback += f"The correct solution is {correct_solution}"
        
        return scor, feedback
    
    
    def evaluate(self, user_answer: str) -> Tuple[float, str]:
        """
        Evaluează răspunsul user-ului pentru întrebarea curentă
        
        Returns:
            (scor 0-100, feedback string)
        """
        if not self.generator.current_metadata:
            return 0.0, "No active question! Ask for a question first."
        
        question_type = self.generator.current_metadata['type']
        
        if question_type == 'n-queens':
            return self.evaluate_n_queens(user_answer)
        else:
            return 0.0, "Scuze momentan inca nu sunt antrenat pe acest tip de problema! \n O sa invat despre {question_type}\n"


class SmarTestSystem:
    """Sistemul (aka chatul)"""
    
    def __init__(self):
        self.generator = QuestionGenerator()
        self.evaluator = AnswerEvaluator(self.generator)
        self.state = 'idle'  # 'idle', 'waiting_answer'
    
    def request_question(self, topic: str, difficulty: str = 'easy') -> str:
        """Functia care returneaza intrebarea ceruta de user pt topicul topic"""
        question = self.generator.generate_question(topic, difficulty)
        if question:
            self.state = 'waiting_answer'
            return question
        else:
            return "Failed to generate question. Try again."
    
    def submit_answer(self, answer: str) -> Dict:
        """Am primit raspunsul de la user"""
        scor, feedback = self.evaluator.evaluate(answer)
        self.state = 'idle'
        return {
            'scor': scor,
            'feedback': feedback,
            'correct_solution': self.generator.current_solution
        }
    
    def chat(self):
        print("-"*70)
        print("SMARTEST")
        print("-"*70)
        print("\nCommands:")
        print("  'question (topic) (dificulty)")
        print("  'hint' pt hinturi")
        print()
        
        while True:
            try:
                user_input = input("USER: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit']:
                    print("SUCCES LA EXAMENE")
                    break
                
                # Parsam comanda
                if user_input.startswith('question'):
                    
                    parts = user_input.split()
                    topic = parts[1] if len(parts) > 1 else 'n-queens'
                    difficulty = parts[2] if len(parts) > 2 else 'easy'
                    
                    question = self.request_question(topic, difficulty)
                    print(f"\nIntrebare ({difficulty}):")
                    print(f"{question}\n")
                
                elif user_input.startswith('answer'):
                    # AICI TOT ce este dupa answer este considerat automat raspuns
                    answer = user_input[7:].strip()
                    
                    result = self.submit_answer(answer)
                    
                    if 'error' in result:
                        print(f"\nX {result['error']}\n")
                    else:
                        print(f"\nscor: {result['scor']:.1f}/100")
                        print(f"{result['feedback']}\n")
                
                elif user_input == 'hint':
                    if self.state == 'waiting_answer':
                        metadata = self.generator.current_metadata
                        if metadata['type'] == 'n-queens':
                            partial = metadata['partial']
                            print(f"\n hint:ai deja :{partial}")
                    else:
                        print("\n⚠️  No active question!\n")
                
                else:
                    if self.state == 'waiting_answer':
                        result = self.submit_answer(user_input)
                        print(f"\nScor: {result['scor']:.1f}/100")
                        print(f"{result['feedback']}\n")
                    else:
                        print("\ncomanda invalida incearca question (topic) (dificultate)")
            
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nX Error: {e}\n")



system = SmarTestSystem()
system.chat()
