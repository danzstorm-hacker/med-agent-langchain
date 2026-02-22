"""
MediAgent - LangGraph state machine

Define el grafo del agente con nodos y edges.
Los nodos con interrupt() pausan automáticamente el grafo.
"""
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    nodo_clasificar_y_sedes,
    nodo_doctores_horarios,
    nodo_confirmar,
    nodo_agendar,
)


def _router_post_sedes(state: AgentState) -> str:
    """Decide a dónde ir después de elegir sede."""
    if state.get("etapa") == "sin_sedes":
        return END
    return "doctores_horarios"


def _router_post_doctores(state: AgentState) -> str:
    """Decide a dónde ir después de elegir doctor+horario."""
    if state.get("etapa") == "sin_doctores":
        return END
    return "confirmar"


def _router_post_confirmar(state: AgentState) -> str:
    """Decide a dónde ir después de la confirmación."""
    if state.get("etapa") == "cancelado":
        return END
    return "agendar"


def build_graph():
    """
    Construye y retorna el grafo compilado con checkpointing.
    
    Flujo:
    START → clasificar_y_sedes (HITL: elige sede)
          → doctores_horarios (HITL: elige doctor+horario)  
          → confirmar (HITL: confirma sí/no)
          → agendar → END
    """
    builder = StateGraph(AgentState)
    
    # ── Agregar nodos ──
    builder.add_node("clasificar_y_sedes", nodo_clasificar_y_sedes)
    builder.add_node("doctores_horarios", nodo_doctores_horarios)
    builder.add_node("confirmar", nodo_confirmar)
    builder.add_node("agendar", nodo_agendar)
    
    # ── Agregar edges ──
    builder.add_edge(START, "clasificar_y_sedes")
    builder.add_conditional_edges("clasificar_y_sedes", _router_post_sedes)
    builder.add_conditional_edges("doctores_horarios", _router_post_doctores)
    builder.add_conditional_edges("confirmar", _router_post_confirmar)
    builder.add_edge("agendar", END)
    
    # ── Compilar con checkpointing ──
    # MemorySaver: estado en memoria (para MVP)
    # Para producción: usar SqliteSaver o PostgresSaver
    checkpointer = MemorySaver()
    
    graph = builder.compile(checkpointer=checkpointer)
    
    return graph


# Singleton del grafo
graph = build_graph()
