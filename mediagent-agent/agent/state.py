"""
MediAgent - State definition for LangGraph
"""
from typing import TypedDict, Optional, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Estado del agente de citas médicas."""
    # Historial de mensajes (LangChain messages)
    messages: Annotated[list, add_messages]
    
    # Datos del paciente logueado
    paciente: Optional[dict]
    
    # Etapa actual del flujo
    etapa: str
    
    # Datos de selección
    sedes_disponibles: Optional[list]
    sede_elegida: Optional[dict]
    doctores_horarios: Optional[list]
    doctor_elegido: Optional[dict]
    horario_elegido: Optional[dict]
    
    # Resultado final
    cita_creada: Optional[dict]
