use clap::Parser;

#[derive(Parser)]
#[command(version, about)]
pub struct CosdataArgs {
    /// The admin key used to encrypt data and sesion tokens.
    #[arg(long)]
    pub admin_key: String,
}
