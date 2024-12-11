
const products = document.querySelectorAll('.product');
const cartItemsList = document.getElementById('cart-items');
const totalElement = document.getElementById('total');
const checkoutButton = document.getElementById('checkout');
let cart = [];
let total = 0;

products.forEach(product => {
    const addToCartButton = product.querySelector('.add-to-cart');
    const minusButton = product.querySelector('.minusBtn');
    const plusButton = product.querySelector('.plusBtn');
    const quantitySpan = product.querySelector('.quantity');
    const productName = product.querySelector('h3').textContent;
    const productPriceText = product.querySelector('p').textContent;

    // Use regular expression to extract the price value
    const productPrice = extractPrice(productPriceText);

    addToCartButton.addEventListener('click', () => {
        const quantity = parseInt(quantitySpan.textContent);
        const itemTotal = productPrice * quantity;

        cart.push({ name: productName, quantity, total: itemTotal });
        total += itemTotal;
        updateCart();
    });

    plusButton.addEventListener('click', () => {
        let quantity = parseInt(quantitySpan.textContent);
        quantity++;
        quantitySpan.textContent = quantity;
    });

    minusButton.addEventListener('click', () => {
        let quantity = parseInt(quantitySpan.textContent);
        if (quantity > 1) {
            quantity--;
            quantitySpan.textContent = quantity;
        }
    });
});

checkoutButton.addEventListener('click', () => {
    alert('Your order has been placed!');
    cart = [];
    total = 0;
    updateCart();
});

function updateCart() {
    cartItemsList.innerHTML = '';
    cart.forEach(item => {
        const listItem = document.createElement('li');
        listItem.textContent = `${item.name} - Quantity: ${item.quantity} - Total: $${item.total.toFixed(2)}`;
        cartItemsList.appendChild(listItem);
    });
    totalElement.textContent = `Total: $${total.toFixed(2)}`;
}
const productPrice = extractPrice(productPriceText);
// Function to extract price from text
function extractPrice(priceText) {
    const priceRegex = /\d+(\.\d{1,2})?/;
    const matches = priceText.match(priceRegex);
    if (matches && matches.length > 0) {
        return parseFloat(matches[0]);
    }
    return 0;
}
function extractPrice(priceText) {
    const priceRegex = /\$\d+(\.\d{1,2})?/; // Regular expression to match currency format ($X.YY)
    const matches = priceText.match(priceRegex);
    if (matches && matches.length > 0) {
        return parseFloat(matches[0].slice(1)); // Remove "$" and convert to float
    }
    return 0;
}