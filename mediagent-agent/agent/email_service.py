"""
MediAgent - Servicio de envío de correos electrónicos (Resend)

Usa Resend (https://resend.com) para enviar correos de confirmación.
Configuración mínima: solo necesitas RESEND_API_KEY en tu .env

Si no tienes dominio verificado, usa:
  EMAIL_FROM=onboarding@resend.dev  (solo envía al correo de tu cuenta Resend)
"""
import os
import resend

# ── Configuración ──
resend.api_key = os.getenv("RESEND_API_KEY", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "MediAgent <onboarding@resend.dev>")


def _build_confirmation_html(
    paciente: dict,
    doctor: dict,
    sede: dict,
    horario: dict,
    especialidad: str,
    fecha_fmt: str,
    cita_id: str,
) -> str:
    """Genera el HTML del correo de confirmación de cita."""
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0; padding:0; background-color:#f4f7fa; font-family: 'Segoe UI', Arial, sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f7fa; padding: 40px 0;">
            <tr>
                <td align="center">
                    <table width="600" cellpadding="0" cellspacing="0" style="background-color:#ffffff; border-radius:12px; overflow:hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
                        
                        <!-- Header -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #0077B6 0%, #00B4D8 100%); padding: 32px 40px; text-align:center;">
                                <h1 style="color:#ffffff; margin:0; font-size:28px; font-weight:700;">🏥 MediAgent</h1>
                                <p style="color:#CAF0F8; margin:8px 0 0; font-size:14px;">Tu asistente de citas médicas</p>
                            </td>
                        </tr>
                        
                        <!-- Success Badge -->
                        <tr>
                            <td style="padding: 32px 40px 16px; text-align:center;">
                                <div style="display:inline-block; background-color:#D4EDDA; color:#155724; padding:10px 24px; border-radius:24px; font-size:16px; font-weight:600;">
                                    ✅ Cita Confirmada
                                </div>
                            </td>
                        </tr>
                        
                        <!-- Greeting -->
                        <tr>
                            <td style="padding: 8px 40px 24px; text-align:center;">
                                <p style="font-size:18px; color:#333; margin:0;">
                                    ¡Hola <strong>{paciente['nombres']}</strong>! Tu cita ha sido agendada exitosamente.
                                </p>
                            </td>
                        </tr>
                        
                        <!-- Appointment Details Card -->
                        <tr>
                            <td style="padding: 0 40px 32px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#F8F9FA; border-radius:10px; border: 1px solid #E9ECEF;">
                                    
                                    <!-- Cita ID -->
                                    <tr>
                                        <td style="padding: 20px 24px 12px;">
                                            <p style="margin:0; font-size:12px; color:#6C757D; text-transform:uppercase; letter-spacing:1px;">Número de cita</p>
                                            <p style="margin:4px 0 0; font-size:18px; color:#0077B6; font-weight:700;">{cita_id}</p>
                                        </td>
                                    </tr>
                                    
                                    <tr><td style="padding:0 24px;"><hr style="border:none; border-top:1px solid #E9ECEF; margin:0;"></td></tr>
                                    
                                    <!-- Doctor -->
                                    <tr>
                                        <td style="padding: 12px 24px;">
                                            <table cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td style="vertical-align:top; padding-right:12px; font-size:20px;">👨‍⚕️</td>
                                                    <td>
                                                        <p style="margin:0; font-size:12px; color:#6C757D;">Doctor</p>
                                                        <p style="margin:2px 0 0; font-size:16px; color:#333; font-weight:600;">Dr(a). {doctor['nombres']} {doctor['apellidos']}</p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                    
                                    <!-- Especialidad -->
                                    <tr>
                                        <td style="padding: 12px 24px;">
                                            <table cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td style="vertical-align:top; padding-right:12px; font-size:20px;">🩺</td>
                                                    <td>
                                                        <p style="margin:0; font-size:12px; color:#6C757D;">Especialidad</p>
                                                        <p style="margin:2px 0 0; font-size:16px; color:#333; font-weight:600;">{especialidad}</p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                    
                                    <!-- Fecha y Hora -->
                                    <tr>
                                        <td style="padding: 12px 24px;">
                                            <table cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td style="vertical-align:top; padding-right:12px; font-size:20px;">📅</td>
                                                    <td>
                                                        <p style="margin:0; font-size:12px; color:#6C757D;">Fecha y hora</p>
                                                        <p style="margin:2px 0 0; font-size:16px; color:#333; font-weight:600;">{fecha_fmt}</p>
                                                        <p style="margin:2px 0 0; font-size:16px; color:#0077B6; font-weight:700;">{horario['hora_inicio']} - {horario['hora_fin']}</p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                    
                                    <!-- Sede -->
                                    <tr>
                                        <td style="padding: 12px 24px 20px;">
                                            <table cellpadding="0" cellspacing="0">
                                                <tr>
                                                    <td style="vertical-align:top; padding-right:12px; font-size:20px;">🏥</td>
                                                    <td>
                                                        <p style="margin:0; font-size:12px; color:#6C757D;">Sede</p>
                                                        <p style="margin:2px 0 0; font-size:16px; color:#333; font-weight:600;">{sede['nombre']}</p>
                                                        <p style="margin:2px 0 0; font-size:14px; color:#6C757D;">📍 {sede['direccion']}</p>
                                                    </td>
                                                </tr>
                                            </table>
                                        </td>
                                    </tr>
                                    
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Reminder -->
                        <tr>
                            <td style="padding: 0 40px 32px;">
                                <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#FFF3CD; border-radius:8px; border: 1px solid #FFEEBA;">
                                    <tr>
                                        <td style="padding: 16px 20px;">
                                            <p style="margin:0; font-size:14px; color:#856404;">
                                                ⏰ <strong>Recuerda:</strong> Llegar 15 minutos antes de tu cita con tu DNI y cualquier examen previo.
                                            </p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        
                        <!-- Footer -->
                        <tr>
                            <td style="background-color:#F8F9FA; padding: 24px 40px; text-align:center; border-top: 1px solid #E9ECEF;">
                                <p style="margin:0; font-size:13px; color:#6C757D;">
                                    Este correo fue enviado automáticamente por MediAgent.<br>
                                    Si no solicitaste esta cita, por favor contáctanos al <strong>01-422-0000</strong>.
                                </p>
                            </td>
                        </tr>
                        
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


def enviar_correo_confirmacion(
    paciente: dict,
    doctor: dict,
    sede: dict,
    horario: dict,
    especialidad: str,
    fecha_fmt: str,
    cita_id: str,
) -> dict:
    """
    Envía un correo de confirmación de cita al paciente usando Resend.

    Returns:
        dict con 'success': bool y 'message': str
    """
    # Verificar API key
    if not resend.api_key:
        return {
            "success": False,
            "message": "RESEND_API_KEY no configurada en .env",
        }

    destinatario = paciente.get("correo")
    if not destinatario:
        return {
            "success": False,
            "message": "El paciente no tiene correo registrado",
        }

    # Construir HTML
    html_content = _build_confirmation_html(
        paciente=paciente,
        doctor=doctor,
        sede=sede,
        horario=horario,
        especialidad=especialidad,
        fecha_fmt=fecha_fmt,
        cita_id=cita_id,
    )

    # Enviar con Resend
    try:
        params: resend.Emails.SendParams = {
            "from": EMAIL_FROM,
            "to": [destinatario],
            "subject": f"✅ Cita Confirmada — {especialidad} | {fecha_fmt}",
            "html": html_content,
        }

        email = resend.Emails.send(params)

        return {
            "success": True,
            "message": f"Correo enviado a {destinatario} (id: {email['id']})",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error al enviar correo con Resend: {str(e)}",
        }
