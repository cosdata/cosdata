use clap::Parser;

#[derive(Parser, Clone)]
#[command(version, about)]
pub struct CosdataArgs {
    /// The admin key used to encrypt data and session tokens.
    #[arg(long)]
    pub admin_key: Option<String>,
    /// Skip confirmation for admin key (not recommended)
    #[arg(long)]
    pub skip_confirmation: bool,
    /// Internal flag to indicate confirmation has been processed
    #[arg(long, hide = true)]
    pub confirmed: bool,
}
