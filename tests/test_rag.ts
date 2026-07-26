import { NextRequest } from 'next/server';
import { POST } from './src/app/api/v1/ml-demo/route';

const testQueries = [
  // 5 Hazır Test Senaryosu (Fixtures from tr.json)
  { type: "Hazır Örnek 1 (Kargo)", text: "Kargom 5 gündür yolda görünüyor ama elime ulaşmadı, ne yapmam lazım?" },
  { type: "Hazır Örnek 2 (İade/Beden)", text: "Aldığım gömleğin bedeni küçük geldi, iade etmek için ne yapabilirim?" },
  { type: "Hazır Örnek 3 (İptal/İade Süresi)", text: "Siparişimi iptal ettim ama üzerinden 3 gün geçti para kartıma yansımadı!" },
  { type: "Hazır Örnek 4 (Hasarlı/Kırık Ürün)", text: "Gelen kargodaki porselen tabak kırık çıktı, tutanağım yok ne olacak?" },
  { type: "Hazır Örnek 5 (Hesap/Şifre SMS)", text: "Şifremi unuttum SMS doğrulama kodu telefona bir türlü gelmiyor giriş yapamıyorum." },

  // 5 Yeni ve Özgün Kullanıcı Senaryosu (New User Queries)
  { type: "Yeni Örnek 1 (Teknik Arıza / Değişim)", text: "Sipariş verdiğim kablosuz kulaklığın sol tarafından ses gelmiyor, garantisi var mı değişim yapmak istiyorum?" },
  { type: "Yeni Örnek 2 (Giriş Sorunu / Şifre Yenileme)", text: "Hesabıma giriş yapmaya çalışırken sürekli hata veriyor, şifre yenileme linki de gelmiyor acil yardım!" },
  { type: "Yeni Örnek 3 (Kampanya / İndirim Kuponu Hatası)", text: "Sepette indirim kuponunu giriyorum ama tutardan düşmüyor, kampanya kodu neden geçersiz hatası alıyorum?" },
  { type: "Yeni Örnek 4 (Şiddetli Şikayet / Olumsuz Duygu)", text: "Müşteri hizmetleriniz o kadar ilgisiz ve kaba ki bir daha asla sizden alışveriş yapmayacağım, rezalet!" },
  { type: "Yeni Örnek 5 (Sipariş İptali / Ücret İadesi)", text: "Dün akşam verdiğim siparişi henüz kargoya verilmeden iptal etmek istiyorum, param ne zaman iade olur?" }
];

async function runTests() {
  console.log("================================================================================");
  console.log("🚀 TÜRKÇE NLU + RAG MOTORU: 10 SENARYOLUK KAPSAMLI DOĞRULAMA TESTİ");
  console.log("================================================================================\n");

  let passed = 0;

  for (let i = 0; i < testQueries.length; i++) {
    const item = testQueries[i];
    console.log(`--- [Senaryo ${i + 1}/10: ${item.type}] ---`);
    console.log(`💬 Soru: "${item.text}"`);

    const req = new NextRequest('http://localhost:3000/api/v1/ml-demo', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: item.text })
    });

    try {
      const res = await POST(req);
      const data = await res.json();

      if (res.status === 200) {
        passed++;
        console.log(`✅ Durum: Başarılı (200 OK - Latency: ${data.meta.latency_ms}ms)`);
        console.log(`🧠 NLU Çıkarımı: Kategori -> [${data.nlu.category}] | Duygu -> [${data.nlu.sentiment}]`);
        console.log(`🎯 RAG Eşleşmesi: [#${data.retrieval.top_policy.id}] "${data.retrieval.top_policy.title}" (Skor: ${data.retrieval.top_policy.match_score})`);
        console.log(`📝 Topraklanmış Yanıt:\n"${data.grounded_response}"\n`);
      } else {
        console.error(`❌ Hata (${res.status}):`, data);
      }
    } catch (err) {
      console.error("❌ İcra Hatası:", err);
    }
  }

  console.log("================================================================================");
  console.log(`📊 SONUÇ: 10 senaryodan ${passed} tanesi başarıyla çalıştı ve yanıt üretti.`);
  console.log("================================================================================");
}

runTests();
