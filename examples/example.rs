use log::{info, warn};
use std::sync::mpsc;
use winit::{
    event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent},
    event_loop::ControlFlow,
};

const APP_VERSION: u32 = 30012024;
const WIN_TITLE: &str = env!("CARGO_PKG_NAME");
const WIN_INIT_WIDTH: u32 = 512;
const WIN_INIT_HEIGHT: u32 = 512;

enum EngineEvent {
    RecreateSurface,
    PauseRendering,
    ResumeRendering,
}

fn main() {
    let (mut engine, event_loop) =
        swain::Engine::new(APP_VERSION, WIN_TITLE, WIN_INIT_WIDTH, WIN_INIT_HEIGHT);

    let (engine_event_tx, engine_event_rx) = mpsc::channel();

    // SECTION : Drawing thread
    let _handle = std::thread::spawn(move || {
        let mut recreate_surface = false;
        let mut rendering = true;

        loop {
            println!("drawing thread");
            while let Ok(event) = engine_event_rx.try_recv() {
                match event {
                    EngineEvent::RecreateSurface => recreate_surface = true,
                    EngineEvent::PauseRendering => rendering = false,
                    EngineEvent::ResumeRendering => rendering = true,
                }
            }
            println!("no events");

            if recreate_surface {
                info!("recreating surface");

                // engine.recreate_surface();

                recreate_surface = false;
            }

            if rendering {
                unsafe {
                    engine.draw();
                }
            }

            let elapsed = engine.last_frame_time.elapsed();
            if elapsed > *swain::DRAW_TIME_MAX {
                warn!(
                    "late by {} ms, restating frame immediately",
                    elapsed.as_millis()
                );
            } else {
                let sleep_duration = *swain::DRAW_TIME_MAX - elapsed;
                std::thread::sleep(sleep_duration);
            }
        }
    });

    // SECTION : Winit event loop
    let mut rendering = true;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                } => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Space),
                            ..
                        },
                    ..
                } => {
                    if rendering {
                        engine_event_tx.send(EngineEvent::PauseRendering).unwrap();
                    } else {
                        engine_event_tx.send(EngineEvent::ResumeRendering).unwrap();
                    }
                    rendering = !rendering;
                }
                WindowEvent::Resized(_) => {
                    engine_event_tx.send(EngineEvent::RecreateSurface).unwrap()
                }
                _ => (),
            },
            _ => (),
        }
    })
}
