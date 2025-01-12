#[derive(Clone, Default)]
pub struct Pagepool<const LEN: usize> {
    inner: Vec<Page<LEN>>,
    current: usize,
}

impl<const LEN: usize> Pagepool<LEN> {
    // pub fn push(&mut self, data: u32) {
    //     // If the pool is empty it creates a new Page and pushes the data to it.
    //     if self.inner.len() == 0 {
    //         let mut page = Page::<LEN>::new();
    //         page.push(data);
    //     }
    //     // If all the current list of chunks are full then create a new one
    //     else if self.current == self.inner.len() - 1 {
    //         let mut page = Page::<LEN>::new();
    //         page.push(data);
    //     }
    //     // If the current chunk is full then go the next chunk
    //     else if self.inner[self.current].is_full() {
    //         self.current += 1;
    //         self.inner[self.current].push(data);
    //     }
    //     // Otherwise push the data to the current chunk
    //     else {
    //         self.inner[self.current].push(data);
    //     }
    // }

    pub fn push(&mut self, data: u32) {
        // If the pool is empty, create the first page and push the data.
        if self.inner.is_empty() {
            let mut page = Page::<LEN>::new();
            page.push(data);
            self.inner.push(page);
            self.current = 0; // Set current to the first page
        }
        // If the current page is full, move to the next one
        else if self.inner[self.current].is_full() {
            // Create a new page and push the data
            let mut page = Page::<LEN>::new();
            page.push(data);
            self.inner.push(page); // Add the new page to the pool
            self.current += 1; // Move to the new page
        }
        // Otherwise, push the data to the current page
        else {
            self.inner[self.current].push(data);
        }
    }

    pub fn contains(&self, data: u32) -> bool {
        self.inner.iter().any(|p| p.data.contains(&data))
    }
}

impl<const LEN: usize> std::ops::Deref for Pagepool<LEN> {
    type Target = Vec<Page<LEN>>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone, PartialEq)]
pub struct Page<const LEN: usize> {
    pub data: [u32; LEN],
    current: usize,
    is_serialized: bool,
}

impl<const LEN: usize> std::ops::Deref for Page<LEN> {
    type Target = [u32; LEN];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<const LEN: usize> Page<LEN> {
    fn new() -> Self {
        Self {
            data: [u32::MAX; LEN],
            is_serialized: false,
            current: 0,
        }
    }

    fn push(&mut self, data: u32) {
        self.current += 1;
        self.data[self.current] = data;
    }
    fn is_full(&self) -> bool {
        self.current == LEN - 1
    }
}

impl<const LEN: usize> AsRef<[u32; LEN]> for Page<LEN> {
    fn as_ref(&self) -> &[u32; LEN] {
        &self.data
    }
}
