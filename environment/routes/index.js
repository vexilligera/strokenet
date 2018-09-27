const puppeteer = require('puppeteer')
const router = require('koa-router')()
const fs = require('fs');

var browser, page

let url = 'file:///' + __dirname + '/../canvas/index.html'
url = url.replace(/\\/g, '/')

router.get('/setsize/:width/:height', async (ctx, next) => {
  let width = parseInt(ctx.params.width)
  let height = parseInt(ctx.params.height)
  browser = await puppeteer.launch({headless: true})
  page = await browser.newPage()
  await page.goto(url)
  await page.setViewport({width: width, height: height});
  await page.evaluate((width, height) => {
    setSize(width, height);
  }, width, height)
  ctx.body = '200'
})

router.get('/setradius/:radius', async (ctx, next) => {
  let radius = parseFloat(ctx.params.radius)
  await page.evaluate((radius) => {
    setRadius(radius);
  }, radius)
})

router.get('/setcolor/:r/:g/:b', async (ctx, next) => {
  let r = parseFloat(ctx.params.r),
  g = parseFloat(ctx.params.g),
  b = parseFloat(ctx.params.b)
  await page.evaluate((r, g, b) => {
    setColor([r, g, b]);
  }, r, g, b)
})

router.post('/stroke', async(ctx, next) => {
  let array = ctx.request.body
  await page.evaluate((array) => {
    stroke(array)
  }, array)
  ctx.body = '200'
})

router.get('/getimage', async(ctx, next) => {
  var image = await page.evaluate(() => {
    return getImage('png');
  })
  ctx.body = image
})

router.get('/close', async(ctx, next) => {
  browser.close()
})

module.exports = router
