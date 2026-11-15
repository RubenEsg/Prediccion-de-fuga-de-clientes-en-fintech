"""La página que se muestra cuando la dirección no corresponde a ninguna sección ni vista."""
import dash

import app
import ui

dash.register_page(__name__, name="No encontrada", title=f"No encontrada · {app.MARCA}")


def layout(**_):
    return ui.pantalla([
        ui.encabezado("Esa página no existe", "La dirección no corresponde a ninguna sección del dash. Estas son las que hay.",
                      pantalla="inicio"),
        ui.seccion("Las secciones y vistas del dash", [ui.destinos([ui.destino(*d) for d in app.DESTINOS])]),
    ])
