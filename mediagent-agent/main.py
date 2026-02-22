"""
MediAgent - Chat de consola para testing

Ejecuta el agente en terminal, manejando el ciclo interrupt/resume
del Human-in-the-Loop de LangGraph.

Uso:
    python main.py
    python main.py --paciente pac-002
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.types import Command

from agent.graph import graph
from agent.tools import get_paciente_by_id, get_especialidad_nombre


# ── Colores para terminal ──
class Colors:
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def print_agent(msg: str):
    """Imprime mensaje del agente."""
    print(f"\n{Colors.GREEN}{Colors.BOLD}🤖 MediAgent:{Colors.RESET}")
    print(f"{Colors.GREEN}{msg}{Colors.RESET}")


def print_system(msg: str):
    """Imprime mensaje del sistema."""
    print(f"{Colors.GRAY}{msg}{Colors.RESET}")


def get_user_input() -> str:
    """Lee input del usuario."""
    return input(f"\n{Colors.BLUE}{Colors.BOLD}👤 Tú: {Colors.RESET}").strip()


def run_chat(paciente_id: str = "pac-001"):
    """
    Ejecuta el chat loop completo.
    
    Flujo:
    1. Carga datos del paciente
    2. Espera que el paciente escriba
    3. Invoca el grafo
    4. Si hay interrupt → muestra mensaje, espera input, resume
    5. Si terminó → muestra mensaje final
    """
    # Cargar paciente
    paciente = get_paciente_by_id(paciente_id)
    if not paciente:
        print(f"Error: Paciente '{paciente_id}' no encontrado.")
        print("Pacientes disponibles: pac-001, pac-002, pac-003, pac-004, pac-005")
        sys.exit(1)
    
    especialidad = get_especialidad_nombre(paciente["especialidad_id"])
    
    print(f"\n{'='*60}")
    print(f"{Colors.CYAN}{Colors.BOLD}  🏥 MediAgent — Chat de Citas Médicas{Colors.RESET}")
    print(f"{'='*60}")
    print_system(f"  Paciente: {paciente['nombres']} {paciente['apellidos']}")
    print_system(f"  Distrito: {paciente['distrito']}")
    print_system(f"  Especialidad: {especialidad}")
    print_system(f"  Enfermedad: {paciente['enfermedad']}")
    print(f"{'='*60}")
    print_system("  Escribe tu mensaje para iniciar. Escribe 'salir' para terminar.\n")
    
    # Config de LangGraph con thread_id único
    config = {"configurable": {"thread_id": f"chat-{paciente_id}"}}
    
    # ── Esperar primer mensaje del paciente ──
    user_input = get_user_input()
    if user_input.lower() in ["salir", "exit", "quit"]:
        print_system("¡Hasta luego! 👋")
        return
    
    # ── Invocar el grafo por primera vez ──
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "paciente": paciente,
        "etapa": "inicio",
    }
    
    # Ejecutar el grafo (se detendrá en el primer interrupt)
    try:
        result = graph.invoke(initial_state, config)
    except Exception as e:
        if "GraphInterrupt" not in str(type(e).__name__):
            raise
    
    # ── Loop principal: manejar interrupts ──
    while True:
        state = graph.get_state(config)
        
        # Si no hay nodos pendientes, el grafo terminó
        if not state.next:
            # Mostrar último mensaje del agente
            if state.values.get("messages"):
                last_ai_msgs = [
                    m for m in state.values["messages"] 
                    if hasattr(m, "type") and m.type == "ai"
                ]
                if last_ai_msgs:
                    print_agent(last_ai_msgs[-1].content)
            print_system("\n[Flujo completado] ✅")
            break
        
        # Hay un interrupt — extraer el mensaje para el usuario
        if state.tasks and state.tasks[0].interrupts:
            interrupt_value = state.tasks[0].interrupts[0].value
            
            if isinstance(interrupt_value, dict) and "message" in interrupt_value:
                print_agent(interrupt_value["message"])
            else:
                print_agent(str(interrupt_value))
        
        # Esperar input del usuario
        user_input = get_user_input()
        if user_input.lower() in ["salir", "exit", "quit"]:
            print_system("¡Hasta luego! 👋")
            break
        
        # Resumir el grafo con la respuesta del usuario
        try:
            result = graph.invoke(Command(resume=user_input), config)
        except Exception as e:
            if "GraphInterrupt" not in str(type(e).__name__):
                raise
    
    print(f"\n{'='*60}")
    print_system("  Sesión terminada. ¡Gracias por usar MediAgent!")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="MediAgent - Chat de consola")
    parser.add_argument(
        "--paciente", 
        default="pac-001",
        help="ID del paciente (default: pac-001). Opciones: pac-001 a pac-005"
    )
    args = parser.parse_args()
    
    # Verificar API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Error: ANTHROPIC_API_KEY no encontrada.")
        print("Crea un archivo .env con: ANTHROPIC_API_KEY=sk-ant-api03-xxx")
        sys.exit(1)
    
    run_chat(args.paciente)


if __name__ == "__main__":
    main()
